
#include <metal_stdlib>
using namespace metal;

struct TropicalParams {
    uint block_len;     
    uint num_states;    
    uint k_bits;        
    uint n_sub;         
    uint sub_size;      
    uint max_block_len; 
    uint tail_bite;     
    uint use_device_cost; 
};

static inline void step_barrier(bool dev) {
    if (dev) {
        threadgroup_barrier(mem_flags::mem_device | mem_flags::mem_threadgroup);
    } else {
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
}

template <typename CostPtr>
static inline void tropical_run(
    device const float*      weights,
    device const float*      levels,
    device uchar*            back,
    constant TropicalParams& p,
    device uchar*            path_out,
    device uint*             init_out,
    CostPtr                  cost_a,
    CostPtr                  cost_b,
    threadgroup uint*        misc,
    uint tid, uint tgsz, uint block_id, bool dev
) {
    const uint num_states = p.num_states;
    const uint k          = p.k_bits;
    const uint blen       = p.block_len;
    const uint n_inputs   = 1u << k;
    const uint n_groups   = num_states >> k;          
    const uint lk_shift   = 31u - clz(num_states) - k; 

    const ulong w_off  = (ulong)block_id * p.max_block_len;
    const ulong lv_off = (ulong)block_id * p.n_sub * num_states;
    const ulong b_off  = (ulong)block_id * p.max_block_len * num_states;

    const uint n_passes = (p.tail_bite != 0u) ? 2u : 1u;
    uint pinned = 0u;

    for (uint pass = 0u; pass < n_passes; pass++) {
        const bool record = (pass + 1u == n_passes);
        const bool pin    = (p.tail_bite != 0u) && record;

        for (uint ns = tid; ns < num_states; ns += tgsz) {
            cost_a[ns] = pin ? ((ns == pinned) ? 0.0f : INFINITY) : 0.0f;
        }
        step_barrier(dev);

        CostPtr cur = cost_a;
        CostPtr nxt = cost_b;

        for (uint step = 0u; step < blen; step++) {
            const float target = weights[w_off + step];
            const ulong lrow   = lv_off + (ulong)(step / p.sub_size) * num_states;
            
            for (uint ns = tid; ns < num_states; ns += tgsz) {
                const float lvl = levels[lrow + ns];
                
                const float d    = target - lvl;
                const float dist = d * d;

                const uint g = ns >> k;   
                float best   = cur[g] + dist;
                uint  best_t = 0u;
                for (uint t = 1u; t < n_inputs; t++) {
                    const float v = cur[g + t * n_groups] + dist;
                    if (v < best) { best = v; best_t = t; }
                }
                nxt[ns] = best;
                if (record) {
                    back[b_off + (ulong)step * num_states + ns] = (uchar)best_t;
                }
            }
            step_barrier(dev);
            CostPtr tmp = cur; cur = nxt; nxt = tmp;
        }

        if (pass == 0u) {
            if (tid == 0u) {
                float bc = INFINITY;
                uint  bs = 0u;
                for (uint s = 0u; s < num_states; s++) {
                    if (cur[s] < bc) { bc = cur[s]; bs = s; }
                }
                misc[0] = bs;
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
            pinned = misc[0];
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }
    }

    threadgroup_barrier(mem_flags::mem_device);

    if (tid == 0u) {
        const uint kmask = n_inputs - 1u;
        uint s = pinned;
        for (uint i = blen; i > 0u; ) {
            i--;
            path_out[w_off + i] = (uchar)(s & kmask);
            const uint t = (uint)back[b_off + (ulong)i * num_states + s];
            s = (s >> k) | (t << lk_shift);
        }
        init_out[block_id] = s;
    }
}

kernel void tropical_encode_block(
    device const float*     weights  [[ buffer(0) ]],
    device const float*     levels   [[ buffer(1) ]],
    device uchar*           back     [[ buffer(2) ]],
    constant TropicalParams* params  [[ buffer(3) ]],
    device uchar*           path_out [[ buffer(4) ]],
    device uint*            init_out [[ buffer(5) ]],
    device float*           dev_cost [[ buffer(6) ]],
    threadgroup float*      cost_a   [[ threadgroup(0) ]],
    threadgroup float*      cost_b   [[ threadgroup(1) ]],
    threadgroup uint*       misc     [[ threadgroup(2) ]],
    uint tid      [[ thread_index_in_threadgroup ]],
    uint tgsz     [[ threads_per_threadgroup ]],
    uint block_id [[ threadgroup_position_in_grid ]]
) {
    constant TropicalParams& p = params[block_id];
    if (p.use_device_cost != 0u) {
        device float* ca = dev_cost + (ulong)block_id * 2u * p.num_states;
        device float* cb = ca + p.num_states;
        tropical_run(weights, levels, back, p, path_out, init_out,
                     ca, cb, misc, tid, tgsz, block_id, true);
    } else {
        tropical_run(weights, levels, back, p, path_out, init_out,
                     cost_a, cost_b, misc, tid, tgsz, block_id, false);
    }
}

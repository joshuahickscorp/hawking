// Decode-step multi-head attention for Qwen-class GQA models.
//
// Computes attention for ONE new token across all heads in a single
// dispatch. Each threadgroup handles one query head; KV head is derived
// via `h / group_size` where `group_size = n_heads / n_kv_heads`.
//
// Layout:
//   - grid: (n_heads, 1, 1)  -- one TG per head
//   - tg size: 64 (power-of-2; required by tree reductions below)
//
// Buffers:
//   0 args  : ArgbufMhaDecode
//   1 q     : (n_heads, head_dim)                f32
//   2 k     : (seq_len, n_kv_heads, head_dim)    f32
//   3 v     : (seq_len, n_kv_heads, head_dim)    f32
//   4 out   : (n_heads, head_dim)                f32
//
// Threadgroup memory layout:
//   shmem[0 .. seq_len]              -- scores buffer (reused for exp(score))
//   shmem[seq_len .. seq_len+tg]     -- reduction scratch (max, then sum)
//
// Capacity: seq_len * 4B + tg_size * 4B threadgroup memory.
// At tg_size=64 and 32 KB shared-mem ceiling, scores cap ~8000 tokens.
// Bench targets (64-256 token generation) fit comfortably.

struct ArgbufMhaDecode {
    uint seq_len;
    uint head_dim;
    uint n_kv_heads;
    uint group_size;     // n_heads / n_kv_heads
    float scale;         // 1 / sqrt(head_dim)
};

kernel void mha_decode_f32(
    constant ArgbufMhaDecode& args   [[buffer(0)]],
    device const float*       q      [[buffer(1)]],
    device const float*       k_cache[[buffer(2)]],
    device const float*       v_cache[[buffer(3)]],
    device       float*       out    [[buffer(4)]],
    threadgroup  float*       shmem  [[threadgroup(0)]],
    uint tg_id   [[threadgroup_position_in_grid]],
    uint tid     [[thread_position_in_threadgroup]],
    uint tg_size [[threads_per_threadgroup]])
{
    const uint h      = tg_id;
    const uint H_DIM  = args.head_dim;
    const uint SEQ    = args.seq_len;
    const uint NKV    = args.n_kv_heads;
    const uint GROUP  = args.group_size;
    const uint kv_h   = h / GROUP;
    const float scale = args.scale;

    threadgroup float* scores = shmem;
    threadgroup float* red    = shmem + SEQ;

    device const float* q_h = q + h * H_DIM;

    // Phase 1: scores[t] = dot(q_h, K[t, kv_h]) * scale
    for (uint t = tid; t < SEQ; t += tg_size) {
        device const float* kt = k_cache + (t * NKV + kv_h) * H_DIM;
        float acc = 0.0f;
        for (uint i = 0; i < H_DIM; ++i) {
            acc += q_h[i] * kt[i];
        }
        scores[t] = acc * scale;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Phase 2: tree-reduce max(scores[0..SEQ])
    float local_max = -INFINITY;
    for (uint t = tid; t < SEQ; t += tg_size) {
        local_max = max(local_max, scores[t]);
    }
    red[tid] = local_max;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint stride = tg_size / 2u; stride > 0u; stride >>= 1) {
        if (tid < stride) {
            red[tid] = max(red[tid], red[tid + stride]);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    float max_score = red[0];
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Phase 3: scores[t] = exp(scores[t] - max); reduce sum
    float local_sum = 0.0f;
    for (uint t = tid; t < SEQ; t += tg_size) {
        float e = exp(scores[t] - max_score);
        scores[t] = e;
        local_sum += e;
    }
    red[tid] = local_sum;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint stride = tg_size / 2u; stride > 0u; stride >>= 1) {
        if (tid < stride) {
            red[tid] += red[tid + stride];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    float inv_sum = 1.0f / red[0];
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Phase 4: out[h, i] = inv_sum * sum_t(scores[t] * V[t, kv_h, i])
    device float* out_h = out + h * H_DIM;
    for (uint i = tid; i < H_DIM; i += tg_size) {
        float acc = 0.0f;
        for (uint t = 0; t < SEQ; ++t) {
            device const float* vt = v_cache + (t * NKV + kv_h) * H_DIM;
            acc += scores[t] * vt[i];
        }
        out_h[i] = acc * inv_sum;
    }
}

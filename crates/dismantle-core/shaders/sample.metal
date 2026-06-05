// sample.metal — wedge 3, on-GPU sampling.
//
// Logits stay on the GPU; only the sampled token id crosses the bus.
//
// Kernels:
//   sample_temperature    — in-place temperature scaling.
//                           [Phase 2.5]
//   sample_repetition     — repetition penalty over a context window.
//                           [Phase 2.5]
//   sample_topk_topp      — fused top-K + top-P + softmax + draw.
//                           Single launch.
//                           [Phase 2.5]
//   sample_argmax_f32     — deterministic greedy argmax over fp32
//                           logits. Bootstrap kernel for token-only
//                           GPU readback.

#include <metal_stdlib>
using namespace metal;

kernel void sample_temperature(
    device       half*  logits [[buffer(0)]],
    constant     uint&  n      [[buffer(1)]],
    constant     float& temp   [[buffer(2)]],
    uint id [[thread_position_in_grid]])
{
    if (id >= n) return;
    if (temp <= 0.0f) return; // greedy: leave untouched
    logits[id] = half((float)logits[id] / temp);
}

kernel void sample_repetition(
    device       half*  logits  [[buffer(0)]],
    device const uint*  recent  [[buffer(1)]],
    constant     uint&  n_recent[[buffer(2)]],
    constant     float& penalty [[buffer(3)]],
    uint id [[thread_position_in_grid]])
{
    if (id >= n_recent) return;
    uint t = recent[id];
    float v = (float)logits[t];
    logits[t] = half(v >= 0.0f ? v / penalty : v * penalty);
}

// v0.5.7-A — parallel 256-thread simdgroup-reduced argmax.
// Same binding scheme as the old serial kernel; grid and tg must both be (256,1,1).
// Two threadgroup buffers: shmem_v (256 floats) and shmem_i (256 uints).
// Tie-breaking: lower index wins (matches CPU reference and old serial path).
kernel void sample_argmax_f32(
    device const float* logits  [[buffer(0)]],
    device       uint*  token   [[buffer(1)]],
    constant     uint&  n       [[buffer(2)]],
    threadgroup  float* shmem_v [[threadgroup(0)]],
    threadgroup  uint*  shmem_i [[threadgroup(1)]],
    uint                tid     [[thread_position_in_threadgroup]],
    uint                tg_size [[threads_per_threadgroup]])
{
    if (n == 0) { if (tid == 0) token[0] = 0; return; }
    float local_v = -INFINITY; uint local_i = 0;
    for (uint i = tid; i < n; i += tg_size) {
        float v = logits[i];
        if (v > local_v) { local_v = v; local_i = i; }
    }
    shmem_v[tid] = local_v; shmem_i[tid] = local_i;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint stride = tg_size / 2u; stride > 0u; stride >>= 1) {
        if (tid < stride) {
            float vb = shmem_v[tid + stride]; uint ib = shmem_i[tid + stride];
            float va = shmem_v[tid]; uint ia = shmem_i[tid];
            if (vb > va || (vb == va && ib < ia)) { shmem_v[tid] = vb; shmem_i[tid] = ib; }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    if (tid == 0) token[0] = shmem_i[0];
}

// Batched greedy argmax: one thread group per slot, 256 threads each.
// Grid: (B * 256, 1, 1). Thread groups: (256, 1, 1).
// Shmem: 256 floats + 256 uints per TG (2 KB per slot).
// Tie-breaking: lower index wins (matches single-slot sample_argmax_f32).
kernel void sample_argmax_f32_batched(
    device const float* logits  [[buffer(0)]],   // (B, vocab) row-major
    device       uint*  tokens  [[buffer(1)]],   // (B,) output token ids
    constant     uint&  n       [[buffer(2)]],   // vocab size
    constant     uint&  batch   [[buffer(3)]],   // B
    threadgroup  float* shmem_v [[threadgroup(0)]],
    threadgroup  uint*  shmem_i [[threadgroup(1)]],
    uint tid     [[thread_position_in_threadgroup]],
    uint tg_size [[threads_per_threadgroup]],
    uint tg_id   [[threadgroup_position_in_grid]])
{
    uint slot = tg_id;
    if (slot >= batch) return;
    device const float* row = logits + (uint64_t)slot * n;
    if (n == 0) { if (tid == 0) tokens[slot] = 0; return; }
    float local_v = -INFINITY; uint local_i = 0;
    for (uint i = tid; i < n; i += tg_size) {
        float v = row[i];
        if (v > local_v) { local_v = v; local_i = i; }
    }
    shmem_v[tid] = local_v; shmem_i[tid] = local_i;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint stride = tg_size / 2u; stride > 0u; stride >>= 1) {
        if (tid < stride) {
            float vb = shmem_v[tid + stride]; uint ib = shmem_i[tid + stride];
            float va = shmem_v[tid]; uint ia = shmem_i[tid];
            if (vb > va || (vb == va && ib < ia)) { shmem_v[tid] = vb; shmem_i[tid] = ib; }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    if (tid == 0) tokens[slot] = shmem_i[0];
}

// v0.5.7-D — parallel top-K selection.
// Finds the K largest logits and writes their values and indices to output buffers.
// K must be ≤ 64. Uses K rounds of parallel argmax; each round excludes
// previously selected indices via a threadgroup "selected" list.
// Threadgroup memory: shmem_v[TG_SIZE], shmem_i[TG_SIZE], selected[64].
#define TOPK_MAX_K 64u
kernel void sample_topk(
    device const float* logits    [[buffer(0)]],
    device       uint*  topk_idx  [[buffer(1)]],
    device       float* topk_val  [[buffer(2)]],
    constant     uint&  n         [[buffer(3)]],
    constant     uint&  k         [[buffer(4)]],
    threadgroup  float* shmem_v   [[threadgroup(0)]],
    threadgroup  uint*  shmem_i   [[threadgroup(1)]],
    threadgroup  uint*  selected  [[threadgroup(2)]],
    uint                tid       [[thread_position_in_threadgroup]],
    uint                tg_size   [[threads_per_threadgroup]])
{
    uint kk = min(k, TOPK_MAX_K);
    for (uint round = 0; round < kk; round++) {
        float local_v = -INFINITY; uint local_i = 0;
        for (uint j = tid; j < n; j += tg_size) {
            bool skip = false;
            for (uint s = 0; s < round; s++) {
                if (selected[s] == j) { skip = true; break; }
            }
            if (!skip) {
                float v = logits[j];
                if (v > local_v) { local_v = v; local_i = j; }
            }
        }
        shmem_v[tid] = local_v; shmem_i[tid] = local_i;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (uint stride = tg_size / 2u; stride > 0u; stride >>= 1) {
            if (tid < stride) {
                float vb = shmem_v[tid + stride]; uint ib = shmem_i[tid + stride];
                float va = shmem_v[tid]; uint ia = shmem_i[tid];
                if (vb > va || (vb == va && ib < ia)) { shmem_v[tid] = vb; shmem_i[tid] = ib; }
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }
        if (tid == 0) {
            topk_idx[round] = shmem_i[0];
            topk_val[round] = shmem_v[0];
            selected[round] = shmem_i[0];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
}

// v0.5.7-E — nucleus (top-P) filtering over top-K outputs.
// Applies temperature, computes normalized cumsum, finds cutoff where cumsum >= top_p.
// surviving_sum = unnormalized exp sum over survivors (with max from survivors).
// This convention is matched by sample_multinomial for a consistent draw.
// Serial single-thread kernel; k ≤ 64.
kernel void sample_topp(
    device const float* topk_val        [[buffer(0)]],
    device const uint*  topk_idx        [[buffer(1)]],
    device       uint*  surviving_count [[buffer(2)]],
    device       float* surviving_sum   [[buffer(3)]],
    constant     uint&  k               [[buffer(4)]],
    constant     float& top_p           [[buffer(5)]],
    constant     float& temperature     [[buffer(6)]],
    uint tid [[thread_position_in_threadgroup]])
{
    if (tid != 0) return;
    uint kk = k;
    // Phase 1: compute max over all k for normalized cumsum
    float max_v_all = -INFINITY;
    for (uint i = 0; i < kk; i++) {
        float v = temperature > 0.0f ? topk_val[i] / temperature : topk_val[i];
        if (v > max_v_all) max_v_all = v;
    }
    float total_exp = 0.0f;
    for (uint i = 0; i < kk; i++) {
        float v = temperature > 0.0f ? topk_val[i] / temperature : topk_val[i];
        total_exp += exp(v - max_v_all);
    }
    // Scan normalized cumsum; find cutoff
    float cumsum = 0.0f;
    uint cutoff = kk;
    for (uint i = 0; i < kk; i++) {
        float v = temperature > 0.0f ? topk_val[i] / temperature : topk_val[i];
        cumsum += exp(v - max_v_all) / total_exp;
        if (cumsum >= top_p) { cutoff = i + 1; break; }
    }
    surviving_count[0] = cutoff;
    // Phase 2: compute surviving_sum = unnorm exp sum with max from survivors.
    // sample_multinomial recomputes max from survivors and uses the same formula.
    float max_v_surv = -INFINITY;
    for (uint i = 0; i < cutoff; i++) {
        float v = temperature > 0.0f ? topk_val[i] / temperature : topk_val[i];
        if (v > max_v_surv) max_v_surv = v;
    }
    float surv_sum = 0.0f;
    for (uint i = 0; i < cutoff; i++) {
        float v = temperature > 0.0f ? topk_val[i] / temperature : topk_val[i];
        surv_sum += exp(v - max_v_surv);
    }
    surviving_sum[0] = surv_sum;
}

// v0.5.7-F — multinomial draw from surviving top-K × top-P distribution.
// uniform_variate is in [0, 1). surviving_sum is the unnormalized exp sum
// of the survivors (as computed by sample_topp). Recomputes max from survivors
// and walks the CDF: cumsum / ss = renormalized probability.
// Serial single-thread kernel; k ≤ 64.
kernel void sample_multinomial(
    device const float* topk_val        [[buffer(0)]],
    device const uint*  topk_idx        [[buffer(1)]],
    device const uint*  surviving_count [[buffer(2)]],
    device const float* surviving_sum   [[buffer(3)]],
    constant     float& uniform_variate [[buffer(4)]],
    device       uint*  out_token       [[buffer(5)]],
    constant     uint&  k               [[buffer(6)]],
    constant     float& temperature     [[buffer(7)]],
    uint tid [[thread_position_in_threadgroup]])
{
    if (tid != 0) return;
    uint sc = surviving_count[0];
    float ss = surviving_sum[0];
    // Recompute max from survivors (same as sample_topp phase 2)
    float max_v = -INFINITY;
    for (uint i = 0; i < sc; i++) {
        float v = temperature > 0.0f ? topk_val[i] / temperature : topk_val[i];
        if (v > max_v) max_v = v;
    }
    float target = uniform_variate * ss;
    float cumsum = 0.0f;
    for (uint i = 0; i < sc; i++) {
        float v = temperature > 0.0f ? topk_val[i] / temperature : topk_val[i];
        cumsum += exp(v - max_v);
        if (cumsum >= target) { out_token[0] = topk_idx[i]; return; }
    }
    out_token[0] = topk_idx[sc > 0 ? sc - 1 : 0];
}


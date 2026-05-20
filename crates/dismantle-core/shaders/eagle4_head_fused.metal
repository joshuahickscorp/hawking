// path-to-100 L5 Lever A — fused rmsnorm + residual-gate output stage of
// the Eagle4 head's per-step forward.
//
// Replaces the CPU tail of `forward_full_metal_no_lm_head_on`:
//   baseline      = rmsnorm(h_high, output_norm, eps)         (HIDDEN,)
//   draft_hidden  = baseline + residual_gate · x              (HIDDEN,)
// where `x` is the head's TCB residual output (post attn + mlp adds) and
// `residual_gate` is either a scalar (length 1, broadcast) or a per-dim
// vector (length HIDDEN). The CPU version lives at
// `crates/dismantle-core/src/speculate/eagle4_head.rs:771-806`.
//
// Why fuse: lets the head's TCB encode the rmsnorm + gate stage AND a
// downstream lm_head argmax (`gemv_f16` + `sample_argmax_f32`, both
// already TCB-friendly) without an intermediate CPU readback. Eliminates
// ONE `commit_and_wait` per chain step in Eagle4 chain spec decode (path-
// to-100 Lever A from phase_l7_2_postmortem.md → L5 carryover).
//
// Geometry:
//   Grid:    (TG_SIZE, 1, 1)     — single TG, classic rmsnorm pattern
//   TG:      (TG_SIZE, 1, 1)     — 256 threads
//   shmem:   TG_SIZE × 4 bytes   — 1 KB partial-sum buffer
//
// V2-Lite HIDDEN=2048 ⇒ each thread processes 8 elements in the variance
// reduction and 8 in the write-out, all coalesced. Latency dominated by
// the threadgroup_barrier sweep in the reduction (log2(256)=8 stages),
// not by the per-element math.

#include <metal_stdlib>
using namespace metal;

// Fused rmsnorm + per-element residual-gate add.
//
//   baseline = h_high · weight / sqrt(mean(h_high²) + eps)
//   out      = baseline + gate · x
//
// `gate_is_vector != 0` selects gate[i]; otherwise gate[0] broadcasts.
// `weight` is the rmsnorm scale (HIDDEN,) — Eagle4's `output_norm`.
// `x` is the HIDDEN-length residual stream from the head's TCB tail.
kernel void eagle4_rmsnorm_residual_gate(
    device const float* h_high          [[buffer(0)]],   // (HIDDEN,)
    device const float* weight          [[buffer(1)]],   // (HIDDEN,) output_norm
    device const float* gate            [[buffer(2)]],   // (1,) or (HIDDEN,) residual_gate
    device const float* x               [[buffer(3)]],   // (HIDDEN,) head residual
    device       float* out             [[buffer(4)]],   // (HIDDEN,) draft_hidden
    constant     uint&  hidden          [[buffer(5)]],
    constant     uint&  gate_is_vector  [[buffer(6)]],
    constant     float& eps             [[buffer(7)]],
    threadgroup  float* shmem           [[threadgroup(0)]],
    uint                tid             [[thread_position_in_threadgroup]],
    uint                tg_size         [[threads_per_threadgroup]])
{
    // ── Phase 1: per-thread sum of h_high² ────────────────────────────────
    float partial = 0.0f;
    for (uint i = tid; i < hidden; i += tg_size) {
        float v = h_high[i];
        partial += v * v;
    }
    shmem[tid] = partial;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // ── Phase 2: tree reduction in shared memory ─────────────────────────
    for (uint stride = tg_size / 2u; stride > 0u; stride >>= 1) {
        if (tid < stride) {
            shmem[tid] += shmem[tid + stride];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    // ── Phase 3: rmsnorm + gate fused write-out ──────────────────────────
    float rms = sqrt(shmem[0] / (float)hidden + eps);
    float inv = 1.0f / rms;
    bool  gv  = (gate_is_vector != 0u);
    for (uint i = tid; i < hidden; i += tg_size) {
        float baseline = h_high[i] * inv * weight[i];
        float alpha    = gv ? gate[i] : gate[0];
        out[i] = baseline + alpha * x[i];
    }
}

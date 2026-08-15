// Qwen3-Coder-Next direct-packed terminal-head kernels.
//
// Registered in metal/mod.rs so the composed hybrid token graph can encode
// final RMSNorm, all-row lm_head, reserved-tail mask, greedy sample, and
// feedback guard.  Encoding is not a generate, HCLI, or TPS claim.
//
// Required command order is fixed:
// final RMSNorm -> all 151936 rows -> tail 151669..151935 mask -> deterministic
// lowest-ID-tie argmax -> feedback guard.  No selected-row shortcut or host
// decoded/BF16 fallback is a valid substitute.

#include <metal_stdlib>
using namespace metal;

constant uint qwen80_terminal_head_hidden = 2048u;
constant uint qwen80_terminal_head_rows = 151936u;
constant uint qwen80_terminal_head_tokenizer_vocab = 151669u;
constant uint qwen80_terminal_head_group = 128u;
constant float qwen80_terminal_head_rms_epsilon = 1.0e-6f;

inline float qwen80_terminal_head_direct_packed_value(
    const device uchar* signs,
    const device half* scales,
    uint element,
    uint group_size)
{
    const uint group = element / group_size;
    const uint in_group = element % group_size;
    const uint sign_byte = group * (group_size / 8u) + in_group / 8u;
    const uchar bit = (signs[sign_byte] >> (in_group % 8u)) & 1u;
    const float scale = float(scales[group]);
    return bit == 1u ? scale : -scale;
}

// One 256-thread group computes RMS(x) then scales x by direct-packed final
// norm values.  Buffers: 0 post-48 hidden f32[2048], 1 norm signs byte[256],
// 2 norm scales half[16], 3 normalized f32[2048], 4 elements, 5 group,
// 6 epsilon.  This is intentionally exact rather than generic.
kernel void qwen80_terminal_head_final_rmsnorm_direct_packed(
    const device float* hidden [[buffer(0)]],
    const device uchar* norm_signs [[buffer(1)]],
    const device half* norm_scales [[buffer(2)]],
    device float* normalized [[buffer(3)]],
    constant uint& elements [[buffer(4)]],
    constant uint& group_size [[buffer(5)]],
    constant float& epsilon [[buffer(6)]],
    uint lane [[thread_position_in_threadgroup]])
{
    if (elements != qwen80_terminal_head_hidden ||
        group_size != qwen80_terminal_head_group ||
        epsilon != qwen80_terminal_head_rms_epsilon) {
        return;
    }
    threadgroup float partial[256];
    float sum = 0.0f;
    for (uint index = lane; index < elements; index += 256u) {
        const float value = hidden[index];
        sum += value * value;
    }
    partial[lane] = sum;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint stride = 128u; stride > 0u; stride >>= 1u) {
        if (lane < stride) partial[lane] += partial[lane + stride];
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    const float inverse_rms = rsqrt(partial[0] / float(elements) + epsilon);
    for (uint index = lane; index < elements; index += 256u) {
        normalized[index] = hidden[index] * inverse_rms *
            qwen80_terminal_head_direct_packed_value(
                norm_signs, norm_scales, index, group_size);
    }
}

// One 256-thread group per lm_head row.  The y coordinate must cover all
// 151936 rows; a future host preflight rejects any selected-row optimization.
// Buffers: 0 head signs, 1 head scales, 2 normalized f32[2048], 3 logits,
// 4 rows, 5 columns, 6 group.
kernel void qwen80_terminal_head_all_row_direct_packed(
    const device uchar* head_signs [[buffer(0)]],
    const device half* head_scales [[buffer(1)]],
    const device float* normalized [[buffer(2)]],
    device float* logits [[buffer(3)]],
    constant uint& rows [[buffer(4)]],
    constant uint& columns [[buffer(5)]],
    constant uint& group_size [[buffer(6)]],
    uint lane [[thread_position_in_threadgroup]],
    uint row [[threadgroup_position_in_grid]])
{
    if (row >= rows || rows != qwen80_terminal_head_rows ||
        columns != qwen80_terminal_head_hidden ||
        group_size != qwen80_terminal_head_group) {
        return;
    }
    threadgroup float partial[256];
    float subtotal = 0.0f;
    const uint row_base = row * columns;
    for (uint column = lane; column < columns; column += 256u) {
        subtotal += qwen80_terminal_head_direct_packed_value(
                        head_signs, head_scales, row_base + column, group_size) *
                    normalized[column];
    }
    partial[lane] = subtotal;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint stride = 128u; stride > 0u; stride >>= 1u) {
        if (lane < stride) partial[lane] += partial[lane + stride];
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    if (lane == 0u) logits[row] = partial[0];
}

// Grid is (256, 267, 1).  Each Y entry maps exactly one reserved token row.
// Buffers: 0 logits, 1 first_reserved, 2 reserved_rows, 3 total_rows.
kernel void qwen80_terminal_head_mask_reserved_tail(
    device float* logits [[buffer(0)]],
    constant uint& first_reserved [[buffer(1)]],
    constant uint& reserved_rows [[buffer(2)]],
    constant uint& total_rows [[buffer(3)]],
    uint lane [[thread_position_in_threadgroup]],
    uint tail_row [[threadgroup_position_in_grid]])
{
    if (first_reserved != qwen80_terminal_head_tokenizer_vocab ||
        reserved_rows != qwen80_terminal_head_rows - qwen80_terminal_head_tokenizer_vocab ||
        total_rows != qwen80_terminal_head_rows || tail_row >= reserved_rows) {
        return;
    }
    if (lane == 0u) logits[first_reserved + tail_row] = -INFINITY;
}

// Single-thread deterministic argmax over the tokenizer-addressable domain.
// Strict `>` retains the smallest token ID when finite logits tie. Buffers:
// 0 masked logits, 1 sampled token u32[1], 2 tokenizer_vocab, 3 total_rows.
kernel void qwen80_terminal_head_greedy_sample_lowest_id(
    const device float* logits [[buffer(0)]],
    device uint* sampled_token [[buffer(1)]],
    constant uint& tokenizer_vocab [[buffer(2)]],
    constant uint& total_rows [[buffer(3)]],
    uint lane [[thread_position_in_grid]])
{
    if (lane != 0u || tokenizer_vocab != qwen80_terminal_head_tokenizer_vocab ||
        total_rows != qwen80_terminal_head_rows) {
        return;
    }
    uint selected_token = 0u;
    float selected_logit = -INFINITY;
    for (uint candidate = 0u; candidate < tokenizer_vocab; ++candidate) {
        const float candidate_logit = logits[candidate];
        if (isfinite(candidate_logit) && candidate_logit > selected_logit) {
            selected_logit = candidate_logit;
            selected_token = candidate;
        }
    }
    sampled_token[0] = selected_token;
}

// Feedback remains host-owned.  This guard emits one only if the selected
// token lies inside the source tokenizer namespace; no tail token may reach a
// next embedding/state update. Buffers: 0 sampled token, 1 feedback guard,
// 2 tokenizer vocab.
kernel void qwen80_terminal_head_feedback_guard(
    const device uint* sampled_token [[buffer(0)]],
    device uint* feedback_guard [[buffer(1)]],
    constant uint& tokenizer_vocab [[buffer(2)]],
    uint lane [[thread_position_in_grid]])
{
    if (lane != 0u || tokenizer_vocab != qwen80_terminal_head_tokenizer_vocab) return;
    feedback_guard[0] = sampled_token[0] < tokenizer_vocab ? 1u : 0u;
}

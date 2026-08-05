// deepseek_v4_p7.metal -- bounded DeepSeek-V4 P7 mHC-FFN device stages.
//
// This source is linked into the shared Metal library for the narrow
// P4B -> P7 -> P6 -> P7 diagnostic join only:
//
//   true four-lane attention HC residual
//       -> hc_ffn_pre / Sinkhorn -> BF16 FFn RMSNorm
//       -> (P6 owns Gate/router/MoE)
//       -> hc_ffn_post -> true four-lane child residual
//
// The caller must bind all buffers from its existing MetalContext. No buffer
// in this file represents a CPU activation handoff, and these stages are not
// an Engine/HCLI/runtime registration or a performance claim.

#include <metal_stdlib>
using namespace metal;

// Widen a BF16 storage word without requiring a device BF16 arithmetic type.
// The authoritative CPU oracle only admits finite inputs/controls for this
// bounded P7 lane; finite RNE conversion is therefore the required grammar.
static inline float dsv4f_p7_bf16_value(ushort bits)
{
    return as_type<float>(((uint)bits) << 16u);
}

static inline ushort dsv4f_p7_bf16_encode_rne(float value)
{
    const uint bits = as_type<uint>(value);
    const uint low_lsb = (bits >> 16u) & 1u;
    return (ushort)((bits + 0x7fffu + low_lsb) >> 16u);
}

// Source `Block.hc_pre` for the FFN branch.  Unlike the P3A BOS precursor,
// input is a real, lane-major `[4, hidden]` residual state; no lane is
// replicated implicitly.  The scalar order intentionally follows the source
// CPU oracle: lane-major sum-square, row-major 24x16384 linear, initial
// softmax/column pass, 19 more Sinkhorn row/column passes, then lane-order
// reduction and one BF16 store per hidden feature.
#pragma clang fp contract(off)
#pragma clang fp reassociate(off)
kernel void deepseek_v4_p7_mhc_ffn_pre_authority(
    device const ushort* residual_hc_bf16 [[buffer(0)]], // [hc_mult, hidden]
    device const float* hc_fn              [[buffer(1)]], // [mix_width, hc_mult * hidden]
    device const float* hc_scale           [[buffer(2)]], // [3]
    device const float* hc_base            [[buffer(3)]], // [mix_width]
    device       ushort* reduced_bf16      [[buffer(4)]], // [hidden]
    device       float* flat_rsqrt_out     [[buffer(5)]], // [1]
    device       float* mixes_out          [[buffer(6)]], // [mix_width]
    device       float* pre_out            [[buffer(7)]], // [hc_mult]
    device       float* post_out           [[buffer(8)]], // [hc_mult]
    device       float* comb_out           [[buffer(9)]], // [hc_mult, hc_mult]
    constant uint& hidden                   [[buffer(10)]],
    constant uint& hc_mult                  [[buffer(11)]],
    constant uint& mix_width                [[buffer(12)]],
    constant uint& sinkhorn_iters           [[buffer(13)]],
    constant float& norm_eps                [[buffer(14)]],
    constant float& hc_eps                  [[buffer(15)]],
    uint thread_id                          [[thread_position_in_grid]])
{
    constexpr uint kHcMult = 4u;
    constexpr uint kMixWidth = 24u;
    constexpr uint kHidden = 4096u;
    if (thread_id != 0u || hidden != kHidden || hc_mult != kHcMult
        || mix_width != kMixWidth || sinkhorn_iters != 20u
        || !(norm_eps > 0.0f) || !(hc_eps > 0.0f)) {
        return;
    }

    const uint flat_width = hc_mult * hidden;
    float mean_square_sum = 0.0f;
    for (uint lane = 0u; lane < hc_mult; ++lane) {
        const ulong lane_base = (ulong)lane * (ulong)hidden;
        for (uint feature = 0u; feature < hidden; ++feature) {
            const float value = dsv4f_p7_bf16_value(
                residual_hc_bf16[lane_base + (ulong)feature]);
            mean_square_sum = mean_square_sum + value * value;
        }
    }
    const float reciprocal = 1.0f / sqrt(mean_square_sum / (float)flat_width + norm_eps);
    flat_rsqrt_out[0] = reciprocal;

    float mixes[kMixWidth];
    for (uint row = 0u; row < mix_width; ++row) {
        float accumulator = 0.0f;
        const ulong row_base = (ulong)row * (ulong)flat_width;
        for (uint lane = 0u; lane < hc_mult; ++lane) {
            const ulong lane_base = (ulong)lane * (ulong)hidden;
            const ulong source_base = row_base + lane_base;
            for (uint feature = 0u; feature < hidden; ++feature) {
                accumulator = accumulator
                    + hc_fn[source_base + (ulong)feature]
                    * dsv4f_p7_bf16_value(
                        residual_hc_bf16[lane_base + (ulong)feature]);
            }
        }
        mixes[row] = accumulator * reciprocal;
        mixes_out[row] = mixes[row];
    }

    float pre[kHcMult];
    float post[kHcMult];
    float comb[kHcMult * kHcMult];
    for (uint lane = 0u; lane < hc_mult; ++lane) {
        const float pre_value = mixes[lane] * hc_scale[0] + hc_base[lane];
        pre[lane] = 1.0f / (1.0f + exp(-pre_value)) + hc_eps;
        const float post_value = mixes[lane + hc_mult] * hc_scale[1]
            + hc_base[lane + hc_mult];
        post[lane] = 2.0f * (1.0f / (1.0f + exp(-post_value)));
        pre_out[lane] = pre[lane];
        post_out[lane] = post[lane];
    }
    for (uint row = 0u; row < hc_mult; ++row) {
        for (uint column = 0u; column < hc_mult; ++column) {
            const uint index = row * hc_mult + column;
            const uint source_index = index + hc_mult * 2u;
            comb[index] = mixes[source_index] * hc_scale[2] + hc_base[source_index];
        }
    }

    // Initial source `softmax(-1) + eps` row pass.
    for (uint row = 0u; row < hc_mult; ++row) {
        const uint start = row * hc_mult;
        float row_max = comb[start];
        for (uint column = 1u; column < hc_mult; ++column) {
            row_max = max(row_max, comb[start + column]);
        }
        float row_sum = 0.0f;
        for (uint column = 0u; column < hc_mult; ++column) {
            const uint index = start + column;
            comb[index] = exp(comb[index] - row_max);
            row_sum = row_sum + comb[index];
        }
        for (uint column = 0u; column < hc_mult; ++column) {
            const uint index = start + column;
            comb[index] = comb[index] / row_sum + hc_eps;
        }
    }

    // The initial source pass already performed its row normalization.  It
    // then performs one column pass followed by 19 row/column passes.
    for (uint column = 0u; column < hc_mult; ++column) {
        float column_sum = 0.0f;
        for (uint row = 0u; row < hc_mult; ++row) {
            column_sum = column_sum + comb[row * hc_mult + column];
        }
        for (uint row = 0u; row < hc_mult; ++row) {
            const uint index = row * hc_mult + column;
            comb[index] = comb[index] / (column_sum + hc_eps);
        }
    }
    for (uint iteration = 1u; iteration < sinkhorn_iters; ++iteration) {
        for (uint row = 0u; row < hc_mult; ++row) {
            const uint start = row * hc_mult;
            float row_sum = 0.0f;
            for (uint column = 0u; column < hc_mult; ++column) {
                row_sum = row_sum + comb[start + column];
            }
            for (uint column = 0u; column < hc_mult; ++column) {
                const uint index = start + column;
                comb[index] = comb[index] / (row_sum + hc_eps);
            }
        }
        for (uint column = 0u; column < hc_mult; ++column) {
            float column_sum = 0.0f;
            for (uint row = 0u; row < hc_mult; ++row) {
                column_sum = column_sum + comb[row * hc_mult + column];
            }
            for (uint row = 0u; row < hc_mult; ++row) {
                const uint index = row * hc_mult + column;
                comb[index] = comb[index] / (column_sum + hc_eps);
            }
        }
    }
    for (uint index = 0u; index < hc_mult * hc_mult; ++index) {
        comb_out[index] = comb[index];
    }

    for (uint feature = 0u; feature < hidden; ++feature) {
        float reduced = 0.0f;
        for (uint lane = 0u; lane < hc_mult; ++lane) {
            reduced = reduced + pre[lane] * dsv4f_p7_bf16_value(
                residual_hc_bf16[(ulong)lane * (ulong)hidden + (ulong)feature]);
        }
        reduced_bf16[feature] = dsv4f_p7_bf16_encode_rne(reduced);
    }
}

// Source `RMSNorm.forward` at the P7 pre->P6 boundary.  It consumes the
// device-produced reduced BF16 row and source BF16 FFn norm weight, retains
// FP32 sum-square/product ordering, then writes the one BF16 P6 input row.
kernel void deepseek_v4_p7_ffn_rmsnorm_bf16_authority(
    device const ushort* input_bf16 [[buffer(0)]], // [hidden]
    device const ushort* weight_bf16 [[buffer(1)]], // [hidden]
    device       ushort* output_bf16 [[buffer(2)]], // [hidden]
    constant uint& width               [[buffer(3)]],
    constant float& eps                [[buffer(4)]],
    uint thread_id                     [[thread_position_in_grid]])
{
    constexpr uint kHidden = 4096u;
    if (thread_id != 0u || width != kHidden || !(eps > 0.0f)) return;
    float sum_square = 0.0f;
    for (uint index = 0u; index < width; ++index) {
        const float value = dsv4f_p7_bf16_value(input_bf16[index]);
        sum_square = sum_square + value * value;
    }
    const float reciprocal = 1.0f / sqrt(sum_square / (float)width + eps);
    for (uint index = 0u; index < width; ++index) {
        const float value = dsv4f_p7_bf16_value(input_bf16[index]);
        const float scale = dsv4f_p7_bf16_value(weight_bf16[index]);
        output_bf16[index] = dsv4f_p7_bf16_encode_rne(value * reciprocal * scale);
    }
}

// Source `Block.hc_post` after the routed/shared MoE output.  P6 owns the
// first input and its routing buffers; this kernel only combines its BF16
// output with the same caller-owned P4B attention HC residual.  Comb row j
// weights residual lane j; comb column k yields child output lane k.
kernel void deepseek_v4_p7_mhc_ffn_post_authority(
    device const ushort* moe_output_bf16       [[buffer(0)]], // [hidden]
    device const ushort* attention_hc_post_bf16 [[buffer(1)]], // [hc_mult, hidden]
    device const float* post                    [[buffer(2)]], // [hc_mult]
    device const float* comb                    [[buffer(3)]], // [hc_mult, hc_mult]
    device       ushort* child_hc_state_bf16    [[buffer(4)]], // [hc_mult, hidden]
    constant uint& hidden                        [[buffer(5)]],
    constant uint& hc_mult                       [[buffer(6)]],
    uint index                                  [[thread_position_in_grid]])
{
    constexpr uint kHidden = 4096u;
    if (hidden != kHidden || hc_mult != 4u) return;
    const uint count = hidden * hc_mult;
    if (index >= count) return;
    const uint output_lane = index / hidden;
    const uint feature = index - output_lane * hidden;
    float value = post[output_lane] * dsv4f_p7_bf16_value(moe_output_bf16[feature]);
    for (uint source_lane = 0u; source_lane < hc_mult; ++source_lane) {
        value = value + comb[source_lane * hc_mult + output_lane]
            * dsv4f_p7_bf16_value(
                attention_hc_post_bf16[(ulong)source_lane * (ulong)hidden + (ulong)feature]);
    }
    child_hc_state_bf16[index] = dsv4f_p7_bf16_encode_rne(value);
}
#pragma clang fp reassociate(on)
#pragma clang fp contract(on)

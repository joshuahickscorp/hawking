// deepseek_v4_mhc_control_exp.metal -- general mHC control-path exp for P4B/P7.
//
// Promotes the previously diagnostic-only Darwin arm64 expf double-double
// reconstruction into a shared, non-trace-specific helper used by production
// mHC control kernels (sigmoid + Sinkhorn softmax). Domain is finite
// x in [-40, 40], which contains the measured mHC control inputs.
//
// This is intentionally general: no fixed-input bit patch, no per-trace
// special case. Callers retain NumericParityV21Only until an end-to-end
// sealed exact-storage receipt is earned for the full composed path.
//
// License note: table/constants reconstruct the active Darwin libSystem expf
// normal path as software float double-double for Metal (no device double).

#include <metal_stdlib>
using namespace metal;

#pragma clang fp contract(off)
#pragma clang fp reassociate(off)

// Auto-derived Darwin arm64 expf double-double table (index-encoded F64 split).
static constant float4 deepseek_v4_mhc_darwin_expf_dd_table[128] = {
    float4(as_type<float>(0x3f800000u), as_type<float>(0x00000000u), as_type<float>(0x00000000u), 0.0f),
    float4(as_type<float>(0x3f80b1edu), as_type<float>(0x331fb333u), as_type<float>(0x26a00000u), 0.0f),
    float4(as_type<float>(0x3f8164d2u), as_type<float>(0xb1c43fd0u), as_type<float>(0x25800000u), 0.0f),
    float4(as_type<float>(0x3f8218afu), as_type<float>(0x3306e7f8u), as_type<float>(0x26a00000u), 0.0f),
    float4(as_type<float>(0x3f82cd87u), as_type<float>(0xb34ea7a9u), as_type<float>(0x26800000u), 0.0f),
    float4(as_type<float>(0x3f838359u), as_type<float>(0x331ddf6eu), as_type<float>(0xa6000000u), 0.0f),
    float4(as_type<float>(0x3f843a29u), as_type<float>(0xb2f14c87u), as_type<float>(0x00000000u), 0.0f),
    float4(as_type<float>(0x3f84f1f6u), as_type<float>(0x332c6f38u), as_type<float>(0x26400000u), 0.0f),
    float4(as_type<float>(0x3f85aac3u), as_type<float>(0x334f9891u), as_type<float>(0xa5800000u), 0.0f),
    float4(as_type<float>(0x3f866491u), as_type<float>(0x3337247fu), as_type<float>(0x26e00000u), 0.0f),
    float4(as_type<float>(0x3f871f62u), as_type<float>(0xb352c2e6u), as_type<float>(0x26000000u), 0.0f),
    float4(as_type<float>(0x3f87db35u), as_type<float>(0x337fed32u), as_type<float>(0xa6a00000u), 0.0f),
    float4(as_type<float>(0x3f88980fu), as_type<float>(0xb37eda4bu), as_type<float>(0x25800000u), 0.0f),
    float4(as_type<float>(0x3f8955eeu), as_type<float>(0x30d86398u), as_type<float>(0x00000000u), 0.0f),
    float4(as_type<float>(0x3f8a14d5u), as_type<float>(0x336a92deu), as_type<float>(0x00000000u), 0.0f),
    float4(as_type<float>(0x3f8ad4c6u), as_type<float>(0x330a58e5u), as_type<float>(0x25800000u), 0.0f),
    float4(as_type<float>(0x3f8b95c2u), as_type<float>(0xb260aba1u), as_type<float>(0xa5800000u), 0.0f),
    float4(as_type<float>(0x3f8c57cau), as_type<float>(0xb2ee6e43u), as_type<float>(0x26000000u), 0.0f),
    float4(as_type<float>(0x3f8d1adfu), as_type<float>(0x3336fcb7u), as_type<float>(0x26a00000u), 0.0f),
    float4(as_type<float>(0x3f8ddf04u), as_type<float>(0x32808b9au), as_type<float>(0x26800000u), 0.0f),
    float4(as_type<float>(0x3f8ea43au), as_type<float>(0xb3697465u), as_type<float>(0xa6c00000u), 0.0f),
    float4(as_type<float>(0x3f8f6a81u), as_type<float>(0x323f3647u), as_type<float>(0x25800000u), 0.0f),
    float4(as_type<float>(0x3f9031dcu), as_type<float>(0x330628cdu), as_type<float>(0x26c00000u), 0.0f),
    float4(as_type<float>(0x3f90fa4du), as_type<float>(0xb3682237u), as_type<float>(0x26c00000u), 0.0f),
    float4(as_type<float>(0x3f91c3d3u), as_type<float>(0x33675624u), as_type<float>(0xa7000000u), 0.0f),
    float4(as_type<float>(0x3f928e72u), as_type<float>(0x337b2a64u), as_type<float>(0xa5800000u), 0.0f),
    float4(as_type<float>(0x3f935a2bu), as_type<float>(0x32bc4f9cu), as_type<float>(0xa6400000u), 0.0f),
    float4(as_type<float>(0x3f9426ffu), as_type<float>(0x31fab1c0u), as_type<float>(0x25800000u), 0.0f),
    float4(as_type<float>(0x3f94f4f0u), as_type<float>(0xb32e0212u), as_type<float>(0x25800000u), 0.0f),
    float4(as_type<float>(0x3f95c3ffu), as_type<float>(0xb3725267u), as_type<float>(0x00000000u), 0.0f),
    float4(as_type<float>(0x3f96942du), as_type<float>(0x32dc8061u), as_type<float>(0x26400000u), 0.0f),
    float4(as_type<float>(0x3f97657du), as_type<float>(0x3313e2f5u), as_type<float>(0x26c00000u), 0.0f),
    float4(as_type<float>(0x3f9837f0u), as_type<float>(0x33231b71u), as_type<float>(0x26a00000u), 0.0f),
    float4(as_type<float>(0x3f990b88u), as_type<float>(0xb26cc9f4u), as_type<float>(0x25800000u), 0.0f),
    float4(as_type<float>(0x3f99e046u), as_type<float>(0xb359be90u), as_type<float>(0xa5800000u), 0.0f),
    float4(as_type<float>(0x3f9ab62bu), as_type<float>(0xb0dac01eu), as_type<float>(0x00000000u), 0.0f),
    float4(as_type<float>(0x3f9b8d3au), as_type<float>(0xb30c5563u), as_type<float>(0xa6a00000u), 0.0f),
    float4(as_type<float>(0x3f9c6573u), as_type<float>(0x33505d86u), as_type<float>(0x26c00000u), 0.0f),
    float4(as_type<float>(0x3f9d3edau), as_type<float>(0xb331a601u), as_type<float>(0x26e00000u), 0.0f),
    float4(as_type<float>(0x3f9e196eu), as_type<float>(0x3244ea39u), as_type<float>(0x25800000u), 0.0f),
    float4(as_type<float>(0x3f9ef532u), as_type<float>(0x33412342u), as_type<float>(0x26000000u), 0.0f),
    float4(as_type<float>(0x3f9fd228u), as_type<float>(0x32959004u), as_type<float>(0xa6800000u), 0.0f),
    float4(as_type<float>(0x3fa0b051u), as_type<float>(0x31fb9715u), as_type<float>(0x00000000u), 0.0f),
    float4(as_type<float>(0x3fa18fafu), as_type<float>(0xb2d5eaedu), as_type<float>(0xa5800000u), 0.0f),
    float4(as_type<float>(0x3fa27043u), as_type<float>(0x30c3125au), as_type<float>(0x00000000u), 0.0f),
    float4(as_type<float>(0x3fa3520fu), as_type<float>(0x3351d005u), as_type<float>(0x26e00000u), 0.0f),
    float4(as_type<float>(0x3fa43516u), as_type<float>(0xb323ec33u), as_type<float>(0x00000000u), 0.0f),
    float4(as_type<float>(0x3fa51958u), as_type<float>(0xb37282c2u), as_type<float>(0xa6000000u), 0.0f),
    float4(as_type<float>(0x3fa5fed7u), as_type<float>(0xb32c9d5eu), as_type<float>(0x26e00000u), 0.0f),
    float4(as_type<float>(0x3fa6e595u), as_type<float>(0xb2c0445eu), as_type<float>(0xa6400000u), 0.0f),
    float4(as_type<float>(0x3fa7cd94u), as_type<float>(0xb3162d36u), as_type<float>(0x26e00000u), 0.0f),
    float4(as_type<float>(0x3fa8b6d5u), as_type<float>(0x3233d990u), as_type<float>(0x26000000u), 0.0f),
    float4(as_type<float>(0x3fa9a15bu), as_type<float>(0xb3162b08u), as_type<float>(0x26000000u), 0.0f),
    float4(as_type<float>(0x3faa8d26u), as_type<float>(0x3325d921u), as_type<float>(0xa5800000u), 0.0f),
    float4(as_type<float>(0x3fab7a3au), as_type<float>(0xb314ad82u), as_type<float>(0xa6c00000u), 0.0f),
    float4(as_type<float>(0x3fac6897u), as_type<float>(0xb3368380u), as_type<float>(0xa6400000u), 0.0f),
    float4(as_type<float>(0x3fad583fu), as_type<float>(0xb22deaf6u), as_type<float>(0x25800000u), 0.0f),
    float4(as_type<float>(0x3fae4934u), as_type<float>(0x3325946bu), as_type<float>(0x26e00000u), 0.0f),
    float4(as_type<float>(0x3faf3b79u), as_type<float>(0xb3252decu), as_type<float>(0x27000000u), 0.0f),
    float4(as_type<float>(0x3fb02f0eu), as_type<float>(0xb2d1247fu), as_type<float>(0x25800000u), 0.0f),
    float4(as_type<float>(0x3fb123f6u), as_type<float>(0xb37c5aa8u), as_type<float>(0x26a00000u), 0.0f),
    float4(as_type<float>(0x3fb21a32u), as_type<float>(0xb33333ceu), as_type<float>(0x00000000u), 0.0f),
    float4(as_type<float>(0x3fb311c4u), as_type<float>(0x32154889u), as_type<float>(0x25800000u), 0.0f),
    float4(as_type<float>(0x3fb40aafu), as_type<float>(0xb33b3569u), as_type<float>(0x26400000u), 0.0f),
    float4(as_type<float>(0x3fb504f3u), as_type<float>(0x32cfe77au), as_type<float>(0xa6400000u), 0.0f),
    float4(as_type<float>(0x3fb60094u), as_type<float>(0xb32f4254u), as_type<float>(0xa5800000u), 0.0f),
    float4(as_type<float>(0x3fb6fd92u), as_type<float>(0xb266b974u), as_type<float>(0xa5800000u), 0.0f),
    float4(as_type<float>(0x3fb7fbf0u), as_type<float>(0xb2d5cd70u), as_type<float>(0x26800000u), 0.0f),
    float4(as_type<float>(0x3fb8fbafu), as_type<float>(0x330ec5f7u), as_type<float>(0x26800000u), 0.0f),
    float4(as_type<float>(0x3fb9fcd2u), as_type<float>(0x330a5817u), as_type<float>(0x26800000u), 0.0f),
    float4(as_type<float>(0x3fbaff5bu), as_type<float>(0xb31bd983u), as_type<float>(0xa6e00000u), 0.0f),
    float4(as_type<float>(0x3fbc034au), as_type<float>(0x337de5d4u), as_type<float>(0xa5800000u), 0.0f),
    float4(as_type<float>(0x3fbd08a4u), as_type<float>(0xb3414fe8u), as_type<float>(0x26e00000u), 0.0f),
    float4(as_type<float>(0x3fbe0f68u), as_type<float>(0x31986099u), as_type<float>(0x00000000u), 0.0f),
    float4(as_type<float>(0x3fbf179au), as_type<float>(0xb3130b1au), as_type<float>(0x26000000u), 0.0f),
    float4(as_type<float>(0x3fc0213bu), as_type<float>(0xb33c1e5fu), as_type<float>(0x26000000u), 0.0f),
    float4(as_type<float>(0x3fc12c4du), as_type<float>(0xb2d6663eu), as_type<float>(0x26400000u), 0.0f),
    float4(as_type<float>(0x3fc238d2u), as_type<float>(0x32c478f6u), as_type<float>(0xa6400000u), 0.0f),
    float4(as_type<float>(0x3fc346cdu), as_type<float>(0xb2976da2u), as_type<float>(0xa6400000u), 0.0f),
    float4(as_type<float>(0x3fc4563fu), as_type<float>(0xb2ceb32du), as_type<float>(0x25800000u), 0.0f),
    float4(as_type<float>(0x3fc5672au), as_type<float>(0x320aa837u), as_type<float>(0xa5800000u), 0.0f),
    float4(as_type<float>(0x3fc67991u), as_type<float>(0xb314abb7u), as_type<float>(0xa6800000u), 0.0f),
    float4(as_type<float>(0x3fc78d75u), as_type<float>(0xb2dd5119u), as_type<float>(0xa6000000u), 0.0f),
    float4(as_type<float>(0x3fc8a2d8u), as_type<float>(0x33391ffcu), as_type<float>(0x26c00000u), 0.0f),
    float4(as_type<float>(0x3fc9b9beu), as_type<float>(0xb37323a2u), as_type<float>(0x26a00000u), 0.0f),
    float4(as_type<float>(0x3fcad226u), as_type<float>(0x333c8521u), as_type<float>(0xa5800000u), 0.0f),
    float4(as_type<float>(0x3fcbec15u), as_type<float>(0xb006c6c0u), as_type<float>(0x00000000u), 0.0f),
    float4(as_type<float>(0x3fcd078cu), as_type<float>(0xb3735f84u), as_type<float>(0xa6c00000u), 0.0f),
    float4(as_type<float>(0x3fce248cu), as_type<float>(0x3228fc24u), as_type<float>(0x00000000u), 0.0f),
    float4(as_type<float>(0x3fcf4319u), as_type<float>(0xb2c39b9cu), as_type<float>(0x26400000u), 0.0f),
    float4(as_type<float>(0x3fd06334u), as_type<float>(0xb2944353u), as_type<float>(0xa6400000u), 0.0f),
    float4(as_type<float>(0x3fd184dfu), as_type<float>(0x3344a2d3u), as_type<float>(0x26400000u), 0.0f),
    float4(as_type<float>(0x3fd2a81eu), as_type<float>(0xb35c1daau), as_type<float>(0xa6400000u), 0.0f),
    float4(as_type<float>(0x3fd3ccf1u), as_type<float>(0xb34cf4cau), as_type<float>(0xa7000000u), 0.0f),
    float4(as_type<float>(0x3fd4f35bu), as_type<float>(0xb3286024u), as_type<float>(0xa5800000u), 0.0f),
    float4(as_type<float>(0x3fd61b5eu), as_type<float>(0xb0303218u), as_type<float>(0x00000000u), 0.0f),
    float4(as_type<float>(0x3fd744fdu), as_type<float>(0xb2d4a58au), as_type<float>(0xa6400000u), 0.0f),
    float4(as_type<float>(0x3fd87039u), as_type<float>(0x3318db66u), as_type<float>(0x26c00000u), 0.0f),
    float4(as_type<float>(0x3fd99d16u), as_type<float>(0xb2f61d41u), as_type<float>(0x26400000u), 0.0f),
    float4(as_type<float>(0x3fdacb94u), as_type<float>(0x335e5594u), as_type<float>(0xa6c00000u), 0.0f),
    float4(as_type<float>(0x3fdbfbb8u), as_type<float>(0xb3504a1cu), as_type<float>(0x26e00000u), 0.0f),
    float4(as_type<float>(0x3fdd2d82u), as_type<float>(0xb375ef9bu), as_type<float>(0xa6c00000u), 0.0f),
    float4(as_type<float>(0x3fde60f5u), as_type<float>(0xb37b43e3u), as_type<float>(0x26000000u), 0.0f),
    float4(as_type<float>(0x3fdf9613u), as_type<float>(0xb2851c3fu), as_type<float>(0x25800000u), 0.0f),
    float4(as_type<float>(0x3fe0ccdfu), as_type<float>(0xb21eab59u), as_type<float>(0x00000000u), 0.0f),
    float4(as_type<float>(0x3fe2055bu), as_type<float>(0xac3e1800u), as_type<float>(0x00000000u), 0.0f),
    float4(as_type<float>(0x3fe33f89u), as_type<float>(0x33657d15u), as_type<float>(0xa6a00000u), 0.0f),
    float4(as_type<float>(0x3fe47b6du), as_type<float>(0xb33f9185u), as_type<float>(0x26a00000u), 0.0f),
    float4(as_type<float>(0x3fe5b907u), as_type<float>(0xb2441be6u), as_type<float>(0x25800000u), 0.0f),
    float4(as_type<float>(0x3fe6f85bu), as_type<float>(0xb32a23c0u), as_type<float>(0xa6c00000u), 0.0f),
    float4(as_type<float>(0x3fe8396au), as_type<float>(0x33207898u), as_type<float>(0xa6800000u), 0.0f),
    float4(as_type<float>(0x3fe97c38u), as_type<float>(0x3300d89fu), as_type<float>(0x26000000u), 0.0f),
    float4(as_type<float>(0x3feac0c7u), as_type<float>(0xb24116deu), as_type<float>(0xa5800000u), 0.0f),
    float4(as_type<float>(0x3fec0719u), as_type<float>(0xb31367c6u), as_type<float>(0xa7000000u), 0.0f),
    float4(as_type<float>(0x3fed4f30u), as_type<float>(0x3276cca1u), as_type<float>(0x25800000u), 0.0f),
    float4(as_type<float>(0x3fee9910u), as_type<float>(0xb34fe4bau), as_type<float>(0x00000000u), 0.0f),
    float4(as_type<float>(0x3fefe4bau), as_type<float>(0xb348464au), as_type<float>(0xa5800000u), 0.0f),
    float4(as_type<float>(0x3ff13231u), as_type<float>(0xb330a5edu), as_type<float>(0xa6e00000u), 0.0f),
    float4(as_type<float>(0x3ff28177u), as_type<float>(0x32f167ffu), as_type<float>(0xa6000000u), 0.0f),
    float4(as_type<float>(0x3ff3d290u), as_type<float>(0xb2871670u), as_type<float>(0x26400000u), 0.0f),
    float4(as_type<float>(0x3ff5257du), as_type<float>(0x32292436u), as_type<float>(0x26000000u), 0.0f),
    float4(as_type<float>(0x3ff67a41u), as_type<float>(0x3358e67fu), as_type<float>(0x25800000u), 0.0f),
    float4(as_type<float>(0x3ff7d0dfu), as_type<float>(0x336615a2u), as_type<float>(0x26e00000u), 0.0f),
    float4(as_type<float>(0x3ff9295au), as_type<float>(0xb3094457u), as_type<float>(0x26e00000u), 0.0f),
    float4(as_type<float>(0x3ffa83b3u), as_type<float>(0xb2923758u), as_type<float>(0x00000000u), 0.0f),
    float4(as_type<float>(0x3ffbdfedu), as_type<float>(0x3359cbe1u), as_type<float>(0x26800000u), 0.0f),
    float4(as_type<float>(0x3ffd3e0cu), as_type<float>(0x31cf486cu), as_type<float>(0x00000000u), 0.0f),
    float4(as_type<float>(0x3ffe9e11u), as_type<float>(0x3338f71fu), as_type<float>(0x25800000u), 0.0f)
};

inline float2 deepseek_v4_mhc_dd_renorm(float high, float low)
{
    const float sum = high + low;
    return float2(sum, low - (sum - high));
}

inline float2 deepseek_v4_mhc_dd_add(float2 left, float2 right)
{
    const float sum = left.x + right.x;
    const float virtual_right = sum - left.x;
    float error = (left.x - (sum - virtual_right)) + (right.x - virtual_right);
    error = error + left.y;
    error = error + right.y;
    return deepseek_v4_mhc_dd_renorm(sum, error);
}

inline float2 deepseek_v4_mhc_dd_add_float(float2 left, float right)
{
    return deepseek_v4_mhc_dd_add(left, float2(right, 0.0f));
}

inline float2 deepseek_v4_mhc_dd_mul(float2 left, float2 right)
{
    const float product = left.x * right.x;
    float error = fma(left.x, right.x, -product);
    error = error + left.x * right.y;
    error = error + left.y * right.x;
    error = error + left.y * right.y;
    return deepseek_v4_mhc_dd_renorm(product, error);
}

inline float2 deepseek_v4_mhc_dd_mul_float(float2 left, float right)
{
    const float product = left.x * right;
    const float error = fma(left.x, right, -product) + left.y * right;
    return deepseek_v4_mhc_dd_renorm(product, error);
}

inline int deepseek_v4_mhc_dd_nearest_i32(float2 value)
{
    float rounded_high = rint(value.x);
    int rounded = int(rounded_high);
    const float residual = (value.x - rounded_high) + value.y;
    if (residual > 0.5f || (residual == 0.5f && (rounded & 1) != 0)) {
        rounded += 1;
    } else if (residual < -0.5f || (residual == -0.5f && (rounded & 1) != 0)) {
        rounded -= 1;
    }
    return rounded;
}

// Production mHC control exp: general finite domain, embedded table, no
// caller-bound buffer and no fixed-input patch.
inline float deepseek_v4_mhc_control_expf(float x)
{
    const uint absolute_bits = as_type<uint>(x) & 0x7fffffffu;
    if (absolute_bits > 0x42200000u || !isfinite(x)) {
        return as_type<float>(0x7fc00000u);
    }

    // Exact F32 split of Darwin binary64 128/ln(2):
    // 0x40671547652b82fe = hi + low.
    const float2 inv_ln2_x128 = float2(
        as_type<float>(0x4338aa3bu),
        as_type<float>(0x36257060u));
    const float2 linear = float2(
        as_type<float>(0x3bb17223u),
        as_type<float>(0xaf41ef25u));
    const float2 quadratic = float2(
        as_type<float>(0x3775fdf0u),
        as_type<float>(0xa8cf29aau));

    const float2 reduced_product = deepseek_v4_mhc_dd_mul_float(inv_ln2_x128, x);
    const int n = deepseek_v4_mhc_dd_nearest_i32(reduced_product);
    const float2 remainder = deepseek_v4_mhc_dd_add_float(reduced_product, -float(n));
    const int table_index = n & 127;
    const int exponent = n >= 0 ? n / 128 : -((-n + 127) / 128);
    const float4 source_scale = deepseek_v4_mhc_darwin_expf_dd_table[table_index];
    const float2 scale = float2(
        ldexp(source_scale.x, exponent),
        ldexp(source_scale.y, exponent));
    const float2 correction = deepseek_v4_mhc_dd_add(
        deepseek_v4_mhc_dd_mul(quadratic, remainder), linear);
    const float2 quadratic_term = deepseek_v4_mhc_dd_mul(correction, remainder);
    const float2 output = deepseek_v4_mhc_dd_mul(
        scale,
        deepseek_v4_mhc_dd_add_float(quadratic_term, 1.0f));
    return output.x + output.y;
}

#pragma clang fp reassociate(on)
#pragma clang fp contract(on)

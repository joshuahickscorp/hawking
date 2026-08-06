//! Diagnostic-only search for a general P4B/P7 mHC CPU-exp compatibility path.
//!
//! It first establishes the actual host target: Rust `f32::exp()` versus
//! Darwin `expf`, then compares a source-attributed FDLIBM/FreeBSD expf
//! range-reduction polynomial across a bounded, representative mHC control
//! domain. No Metal/runtime path, authority kernel, artifact, or receipt is
//! modified by this program.

use hawking_core::gravity_deepseek_v4::DeepSeekV4FullStreamReader;
use hawking_core::gravity_deepseek_v4_layer0_continuation::layer0_position1_complete_attention_cpu_oracle;
use hawking_core::gravity_deepseek_v4_layer0_prefix::{
    HC_EPS, HC_MULT, LAYER0_HC_ATTN_BASE, LAYER0_HC_ATTN_SCALE,
};
use serde_json::{json, Value};
use std::error::Error;
use std::path::PathBuf;

#[cfg(target_os = "macos")]
use metal::objc::{msg_send, sel, sel_impl};
#[cfg(target_os = "macos")]
use metal::{CompileOptions, Device, MTLCommandBufferStatus, MTLResourceOptions, MTLSize};

type ProbeResult<T> = Result<T, Box<dyn Error>>;

// `f32::exp` is an intrinsic with platform/version-dependent precision. On
// this macOS arm64 process, this direct ABI probe determines whether it is
// bit-identical to Darwin libSystem's scalar expf for the tested domain.
unsafe extern "C" {
    fn expf(x: f32) -> f32;
}

const HALF: [f32; 2] = [0.5, -0.5];
const LN2_HI: f32 = 6.931_457_5e-1;
const LN2_LO: f32 = 1.428_606_8e-6;
const INV_LN2: f32 = 1.442_695_0;
const P1: f32 = 1.666_662_5e-1;
const P2: f32 = -2.766_733_3e-3;
const DOMAIN_MIN: f32 = -40.0;
const DOMAIN_MAX: f32 = 40.0;

// This is a reconstruction of the arm64 Darwin libSystem `expf` *normal
// finite* fast path observed in the active system image.  It is intentionally
// diagnostic-only: the constants and table are bound to that host library
// image, not presented as a portable CPU-exp contract.  The target uses F64
// range reduction (`128 / ln(2)`), a 128-entry bit table, and a degree-2
// correction before a single F64 -> F32 rounding.
const DARWIN_EXPF_INV_LN2_X128_BITS: u64 = 0x4067_1547_652b_82fe;
const DARWIN_EXPF_MAGIC_BITS: u64 = 0x4338_0000_0000_0000;
const DARWIN_EXPF_POLY_QUADRATIC_BITS: u64 = 0x3eee_bfbd_ff30_d656;
const DARWIN_EXPF_POLY_LINEAR_BITS: u64 = 0x3f76_2e44_53e1_0dae;
const DARWIN_EXPF_TABLE: [u64; 128] = [
    0x3ff0000000000000,
    0x3feff63da9fb3335,
    0x3fefec9a3e778061,
    0x3fefe315e86e7f85,
    0x3fefd9b0d3158574,
    0x3fefd06b29ddf6de,
    0x3fefc74518759bc8,
    0x3fefbe3ecac6f383,
    0x3fefb5586cf9890f,
    0x3fefac922b7247f7,
    0x3fefa3ec32d3d1a2,
    0x3fef9b66affed31b,
    0x3fef9301d0125b51,
    0x3fef8abdc06c31cc,
    0x3fef829aaea92de0,
    0x3fef7a98c8a58e51,
    0x3fef72b83c7d517b,
    0x3fef6af9388c8dea,
    0x3fef635beb6fcb75,
    0x3fef5be084045cd4,
    0x3fef54873168b9aa,
    0x3fef4d5022fcd91d,
    0x3fef463b88628cd6,
    0x3fef3f49917ddc96,
    0x3fef387a6e756238,
    0x3fef31ce4fb2a63f,
    0x3fef2b4565e27cdd,
    0x3fef24dfe1f56381,
    0x3fef1e9df51fdee1,
    0x3fef187fd0dad990,
    0x3fef1285a6e4030b,
    0x3fef0cafa93e2f56,
    0x3fef06fe0a31b715,
    0x3fef0170fc4cd831,
    0x3feefc08b26416ff,
    0x3feef6c55f929ff1,
    0x3feef1a7373aa9cb,
    0x3feeecae6d05d866,
    0x3feee7db34e59ff7,
    0x3feee32dc313a8e5,
    0x3feedea64c123422,
    0x3feeda4504ac801c,
    0x3feed60a21f72e2a,
    0x3feed1f5d950a897,
    0x3feece086061892d,
    0x3feeca41ed1d0057,
    0x3feec6a2b5c13cd0,
    0x3feec32af0d7d3de,
    0x3feebfdad5362a27,
    0x3feebcb299fddd0d,
    0x3feeb9b2769d2ca7,
    0x3feeb6daa2cf6642,
    0x3feeb42b569d4f82,
    0x3feeb1a4ca5d920f,
    0x3feeaf4736b527da,
    0x3feead12d497c7fd,
    0x3feeab07dd485429,
    0x3feea9268a5946b7,
    0x3feea76f15ad2148,
    0x3feea5e1b976dc09,
    0x3feea47eb03a5585,
    0x3feea34634ccc320,
    0x3feea23882552225,
    0x3feea155d44ca973,
    0x3feea09e667f3bcd,
    0x3feea012750bdabf,
    0x3fee9fb23c651a2f,
    0x3fee9f7df9519484,
    0x3fee9f75e8ec5f74,
    0x3fee9f9a48a58174,
    0x3fee9feb564267c9,
    0x3feea0694fde5d3f,
    0x3feea11473eb0187,
    0x3feea1ed0130c132,
    0x3feea2f336cf4e62,
    0x3feea427543e1a12,
    0x3feea589994cce13,
    0x3feea71a4623c7ad,
    0x3feea8d99b4492ed,
    0x3feeaac7d98a6699,
    0x3feeace5422aa0db,
    0x3feeaf3216b5448c,
    0x3feeb1ae99157736,
    0x3feeb45b0b91ffc6,
    0x3feeb737b0cdc5e5,
    0x3feeba44cbc8520f,
    0x3feebd829fde4e50,
    0x3feec0f170ca07ba,
    0x3feec49182a3f090,
    0x3feec86319e32323,
    0x3feecc667b5de565,
    0x3feed09bec4a2d33,
    0x3feed503b23e255d,
    0x3feed99e1330b358,
    0x3feede6b5579fdbf,
    0x3feee36bbfd3f37a,
    0x3feee89f995ad3ad,
    0x3feeee07298db666,
    0x3feef3a2b84f15fb,
    0x3feef9728de5593a,
    0x3feeff76f2fb5e47,
    0x3fef05b030a1064a,
    0x3fef0c1e904bc1d2,
    0x3fef12c25bd71e09,
    0x3fef199bdd85529c,
    0x3fef20ab5fffd07a,
    0x3fef27f12e57d14b,
    0x3fef2f6d9406e7b5,
    0x3fef3720dcef9069,
    0x3fef3f0b555dc3fa,
    0x3fef472d4a07897c,
    0x3fef4f87080d89f2,
    0x3fef5818dcfba487,
    0x3fef60e316c98398,
    0x3fef69e603db3285,
    0x3fef7321f301b460,
    0x3fef7c97337b9b5f,
    0x3fef864614f5a129,
    0x3fef902ee78b3ff6,
    0x3fef9a51fbc74c83,
    0x3fefa4afa2a490da,
    0x3fefaf482d8e67f1,
    0x3fefba1bee615a27,
    0x3fefc52b376bba97,
    0x3fefd0765b6e4540,
    0x3fefdbfdad9cbe14,
    0x3fefe7c1819e90d8,
    0x3feff3c22b8f71f1,
];

fn failure(message: impl Into<String>) -> Box<dyn Error> {
    std::io::Error::new(std::io::ErrorKind::InvalidData, message.into()).into()
}

fn main() -> ProbeResult<()> {
    let artifact = parse_artifact()?;
    let reader = DeepSeekV4FullStreamReader::admit(&artifact)?;
    let cpu = layer0_position1_complete_attention_cpu_oracle(&reader)?;
    let scale = f32_tensor(&reader, LAYER0_HC_ATTN_SCALE)?;
    let base = f32_tensor(&reader, LAYER0_HC_ATTN_BASE)?;
    let p1_inputs = p1_exp_inputs(&cpu.causal.token1_prefix.hc_mixes_f32, &scale, &base)?;
    let corpus = representative_domain(&p1_inputs);
    let rust: Vec<f32> = corpus.iter().map(|&x| x.exp()).collect();
    let darwin: Vec<f32> = corpus.iter().map(|&x| unsafe { expf(x) }).collect();
    let fdlibm: Vec<f32> = corpus
        .iter()
        .map(|&x| fdlibm_expf_control_domain(x))
        .collect();
    let darwin_disassembly_rising: Vec<f32> = corpus
        .iter()
        .map(|&x| darwin_arm64_expf_reconstructed(x, RemainderSign::AXMinusN))
        .collect();
    let darwin_disassembly_falling: Vec<f32> = corpus
        .iter()
        .map(|&x| darwin_arm64_expf_reconstructed(x, RemainderSign::NMinusAX))
        .collect();
    let rust_sigmoid: Vec<f32> = corpus.iter().map(|&x| sigmoid_rust(x)).collect();
    let fdlibm_sigmoid: Vec<f32> = corpus.iter().map(|&x| sigmoid_fdlibm(x)).collect();
    #[cfg(target_os = "macos")]
    let gpu_strict_math = gpu_strict_math_probe(
        &corpus,
        &p1_inputs,
        &cpu.causal.token1_prefix.hc_mixes_f32,
        &cpu.causal.token1_prefix.hc_post_f32,
        &cpu.causal.token1_prefix.hc_comb_f32,
        &scale,
        &base,
        &rust,
        &fdlibm,
        &rust_sigmoid,
    )?;
    #[cfg(not(target_os = "macos"))]
    let gpu_strict_math = Value::String("unavailable outside macOS Metal".to_owned());
    let report = json!({
        "schema":"hawking.gravity.deepseek_v4.p4b_cpu_exp_compat_diagnostic.v1",
        "status":"DIAGNOSTIC_ONLY_NOT_PROMOTED",
        "artifact_manifest_seal_sha256":reader.manifest_seal_sha256(),
        "host_target":"Rust f32::exp intrinsic on this macOS arm64 process",
        "reference_probe":"Darwin libSystem expf via direct C ABI",
        "candidate":"FDLIBM/FreeBSD expf range reduction + degree-2 correction polynomial, restricted to the defined normal mHC control domain",
        "candidate_license_attribution":"algorithm/constants derived from FreeBSD msun e_expf.c via Rust compiler-builtins libm expf.rs; permissive source attribution retained in the later Metal prototype",
        "domain":{"minimum":DOMAIN_MIN,"maximum":DOMAIN_MAX,"definition":"all current P1 post exponent inputs and softmax row-max deltas, a deterministic 16,385-point grid, adjacent representable values around P1 inputs, and 65,536 deterministic pseudo-random normal samples"},
        "p1_exp_inputs_f32_bits":f32_words(&p1_inputs),
        "p1_control_exactness":{
            "rust_vs_fdlibm_expf":f32_delta(
                &p1_inputs.iter().map(|&x| x.exp()).collect::<Vec<_>>(),
                &p1_inputs.iter().map(|&x| fdlibm_expf_control_domain(x)).collect::<Vec<_>>(),
                &p1_inputs),
            "rust_vs_darwin_arm64_reconstruction_expf":f32_delta(
                &p1_inputs.iter().map(|&x| x.exp()).collect::<Vec<_>>(),
                &p1_inputs.iter().map(|&x| darwin_arm64_expf_reconstructed(x, RemainderSign::AXMinusN)).collect::<Vec<_>>(),
                &p1_inputs),
            "rust_vs_fdlibm_sigmoid":f32_delta(
                &p1_inputs.iter().map(|&x| sigmoid_rust(x)).collect::<Vec<_>>(),
                &p1_inputs.iter().map(|&x| sigmoid_fdlibm(x)).collect::<Vec<_>>(),
                &p1_inputs),
        },
        "samples":corpus.len(),
        "rust_vs_darwin_expf":f32_delta(&rust, &darwin, &corpus),
        "rust_vs_fdlibm_expf":f32_delta(&rust, &fdlibm, &corpus),
        "darwin_arm64_disassembly_reconstruction":{
            "binding":"active arm64 libSystem expf fast path; F64 table/range-reduction reconstruction, not portable API contract",
            "a_x_minus_n":f32_delta(&rust, &darwin_disassembly_rising, &corpus),
            "n_minus_a_x":f32_delta(&rust, &darwin_disassembly_falling, &corpus),
        },
        "rust_vs_fdlibm_sigmoid":f32_delta(&rust_sigmoid, &fdlibm_sigmoid, &corpus),
        "gpu_strict_math":gpu_strict_math,
        "receipt_written":false,
        "promotion":false,
    });
    println!("{}", serde_json::to_string(&report)?);
    Ok(())
}

fn parse_artifact() -> ProbeResult<PathBuf> {
    let mut artifact = None;
    let mut arguments = std::env::args_os().skip(1);
    while let Some(flag) = arguments.next() {
        match flag.to_string_lossy().as_ref() {
            "--artifact" => artifact = arguments.next().map(PathBuf::from),
            other => return Err(failure(format!("unknown argument {other}"))),
        }
    }
    artifact.ok_or_else(|| failure("--artifact is required"))
}

fn f32_tensor(reader: &DeepSeekV4FullStreamReader, name: &str) -> ProbeResult<Vec<f32>> {
    let metadata = reader.tensor_metadata(name)?;
    if metadata.dtype != "F32" || metadata.bytes % 4 != 0 {
        return Err(failure(format!("{name} must remain F32")));
    }
    let bytes = reader.read_verified_full(name, metadata.bytes as usize)?;
    Ok(bytes
        .chunks_exact(4)
        .map(|bytes| f32::from_le_bytes([bytes[0], bytes[1], bytes[2], bytes[3]]))
        .collect())
}

fn p1_exp_inputs(mixes: &[f32], scale: &[f32], base: &[f32]) -> ProbeResult<Vec<f32>> {
    if mixes.len() != 24 || scale.len() != 3 || base.len() != 24 {
        return Err(failure("P1 mHC control geometry changed"));
    }
    let mut inputs = Vec::with_capacity(HC_MULT + HC_MULT * HC_MULT);
    for lane in 0..HC_MULT {
        let post_logit = mixes[lane + HC_MULT] * scale[1] + base[lane + HC_MULT];
        inputs.push(-post_logit);
    }
    let mut comb = [0.0; HC_MULT * HC_MULT];
    for row in 0..HC_MULT {
        for column in 0..HC_MULT {
            let index = row * HC_MULT + column;
            let source = index + 2 * HC_MULT;
            comb[index] = mixes[source] * scale[2] + base[source];
        }
    }
    for row in 0..HC_MULT {
        let start = row * HC_MULT;
        let mut maximum = comb[start];
        for column in 1..HC_MULT {
            maximum = maximum.max(comb[start + column]);
        }
        for column in 0..HC_MULT {
            inputs.push(comb[start + column] - maximum);
        }
    }
    let minimum = inputs.iter().copied().fold(f32::INFINITY, f32::min);
    let maximum = inputs.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    if !minimum.is_finite() || !maximum.is_finite() || minimum < DOMAIN_MIN || maximum > DOMAIN_MAX
    {
        return Err(failure(format!(
            "P1 exponent input escaped declared mHC control domain: [{minimum}, {maximum}]",
        )));
    }
    Ok(inputs)
}

fn representative_domain(p1_inputs: &[f32]) -> Vec<f32> {
    let mut values = vec![DOMAIN_MIN, DOMAIN_MAX, -0.0, 0.0, -1.0, 1.0];
    values.extend_from_slice(p1_inputs);
    for &value in p1_inputs {
        values.push(next_down(value));
        values.push(next_up(value));
    }
    // Dense ordered grid catches range-reduction transitions reproducibly.
    for index in 0..=16_384u32 {
        let fraction = index as f32 / 16_384.0;
        values.push(DOMAIN_MIN + (DOMAIN_MAX - DOMAIN_MIN) * fraction);
    }
    // A fixed LCG adds samples with non-grid mantissas throughout the domain.
    let mut state = 0x4d59_5df4_d0f3_3173u64;
    for _ in 0..65_536 {
        state = state
            .wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(1);
        let unit = ((state >> 40) as u32) as f32 / ((1u32 << 24) as f32);
        values.push(DOMAIN_MIN + (DOMAIN_MAX - DOMAIN_MIN) * unit);
    }
    values
}

// Port of the normal finite path in FreeBSD `e_expf.c` / Rust compiler-builtins
// `libm::expf`. The declared mHC domain [-40, 40] keeps its result normal and
// avoids the source routine's overflow/subnormal special cases.
#[inline(never)]
fn fdlibm_expf_control_domain(mut x: f32) -> f32 {
    debug_assert!(x.is_finite() && (DOMAIN_MIN..=DOMAIN_MAX).contains(&x));
    let mut hx = x.to_bits();
    let sign = (hx >> 31) as i32;
    hx &= 0x7fff_ffff;
    let (k, hi, lo) = if hx > 0x3eb1_7218 {
        let k = if hx > 0x3f85_1592 {
            (INV_LN2 * x + HALF[sign as usize]) as i32
        } else {
            1 - sign - sign
        };
        let kf = k as f32;
        let hi = x - kf * LN2_HI;
        let lo = kf * LN2_LO;
        x = hi - lo;
        (k, hi, lo)
    } else if hx > 0x3900_0000 {
        (0, x, 0.0)
    } else {
        return 1.0 + x;
    };
    let xx = x * x;
    let c = x - xx * (P1 + xx * P2);
    let y = 1.0 + (x * c / (2.0 - c) - lo + hi);
    if k == 0 {
        y
    } else {
        scalbn_normal(y, k)
    }
}

#[derive(Clone, Copy)]
enum RemainderSign {
    AXMinusN,
    NMinusAX,
}

// Rebuilds only the already-admitted finite normal domain.  `mul_add` is
// deliberate: the observed host sequence uses FP64 fused multiply-add/
// subtract operations.  The two remainder directions let the diagnostic
// validate the AArch64 fused-negative-subtract interpretation rather than
// silently assume it.
#[inline(never)]
fn darwin_arm64_expf_reconstructed(x: f32, direction: RemainderSign) -> f32 {
    debug_assert!(x.is_finite() && (DOMAIN_MIN..=DOMAIN_MAX).contains(&x));
    let inv_ln2_x128 = f64::from_bits(DARWIN_EXPF_INV_LN2_X128_BITS);
    let magic = f64::from_bits(DARWIN_EXPF_MAGIC_BITS);
    let linear = f64::from_bits(DARWIN_EXPF_POLY_LINEAR_BITS);
    let quadratic = f64::from_bits(DARWIN_EXPF_POLY_QUADRATIC_BITS);
    let xd = f64::from(x);
    let rounded_magic = inv_ln2_x128.mul_add(xd, magic);
    let rounded_bits = rounded_magic.to_bits();
    let rounded = rounded_magic - magic;
    let remainder = match direction {
        RemainderSign::AXMinusN => inv_ln2_x128.mul_add(xd, -rounded),
        RemainderSign::NMinusAX => (-inv_ln2_x128).mul_add(xd, rounded),
    };
    let scaled_bits =
        DARWIN_EXPF_TABLE[(rounded_bits & 0x7f) as usize].wrapping_add(rounded_bits << 45);
    let scaled = f64::from_bits(scaled_bits);
    let correction = quadratic.mul_add(remainder, linear);
    let quadratic_term = correction * remainder;
    quadratic_term.mul_add(scaled, scaled) as f32
}

#[inline]
fn scalbn_normal(value: f32, exponent_delta: i32) -> f32 {
    let bits = value.to_bits();
    let exponent = ((bits >> 23) & 0xff) as i32 + exponent_delta;
    debug_assert!((1..=254).contains(&exponent));
    f32::from_bits((bits & 0x807f_ffff) | ((exponent as u32) << 23))
}

#[inline]
fn sigmoid_rust(value: f32) -> f32 {
    1.0 / (1.0 + (-value).exp())
}

#[inline]
fn sigmoid_fdlibm(value: f32) -> f32 {
    1.0 / (1.0 + fdlibm_expf_control_domain(-value))
}

fn next_up(value: f32) -> f32 {
    if value.is_nan() || value == f32::INFINITY {
        return value;
    }
    if value == 0.0 {
        return f32::from_bits(1);
    }
    let bits = value.to_bits();
    f32::from_bits(if value > 0.0 { bits + 1 } else { bits - 1 })
}

fn next_down(value: f32) -> f32 {
    if value.is_nan() || value == f32::NEG_INFINITY {
        return value;
    }
    if value == 0.0 {
        return f32::from_bits(0x8000_0001);
    }
    let bits = value.to_bits();
    f32::from_bits(if value > 0.0 { bits - 1 } else { bits + 1 })
}

#[cfg(target_os = "macos")]
fn gpu_strict_math_probe(
    inputs: &[f32],
    p1_inputs: &[f32],
    p1_mixes: &[f32],
    p1_post: &[f32],
    p1_comb: &[f32],
    hc_scale: &[f32],
    hc_base: &[f32],
    rust_exp: &[f32],
    cpu_fdlibm_exp: &[f32],
    rust_sigmoid: &[f32],
) -> ProbeResult<Value> {
    const KERNEL: &str = "deepseek_v4_p4b_fdlibm_expf_compat_domain_candidate";
    const DARWIN_DD_KERNEL: &str = "deepseek_v4_p4b_darwin_expf_dd_compat_domain_candidate";
    let device = Device::system_default().ok_or_else(|| failure("no Metal device"))?;
    let fp64_language_probe = metal_fp64_language_probe(&device);
    let queue = device.new_command_queue();
    let options = CompileOptions::new();
    options.set_fast_math_enabled(false);
    let library = device
        .new_library_with_source(hawking_core::metal::SHADER_MATMUL, &options)
        .map_err(failure)?;
    let function = library.get_function(KERNEL, None).map_err(failure)?;
    let pipeline = device
        .new_compute_pipeline_state_with_function(&function)
        .map_err(failure)?;
    let darwin_dd_function = library
        .get_function(DARWIN_DD_KERNEL, None)
        .map_err(failure)?;
    let darwin_dd_pipeline = device
        .new_compute_pipeline_state_with_function(&darwin_dd_function)
        .map_err(failure)?;
    let input = gpu_f32_buffer(&device, inputs);
    let fdlibm_out = gpu_f32_buffer(&device, &vec![0.0; inputs.len()]);
    let precise_out = gpu_f32_buffer(&device, &vec![0.0; inputs.len()]);
    let sigmoid_out = gpu_f32_buffer(&device, &vec![0.0; inputs.len()]);
    let count = u32::try_from(inputs.len()).map_err(|_| failure("GPU corpus count overflow"))?;
    let command = queue.new_command_buffer();
    let encoder = command.new_compute_command_encoder();
    encoder.set_compute_pipeline_state(&pipeline);
    encoder.set_buffer(0, Some(&input), 0);
    encoder.set_buffer(1, Some(&fdlibm_out), 0);
    encoder.set_buffer(2, Some(&precise_out), 0);
    encoder.set_buffer(3, Some(&sigmoid_out), 0);
    encoder.set_bytes(
        4,
        std::mem::size_of::<u32>() as u64,
        &count as *const u32 as *const _,
    );
    encoder.dispatch_threads(
        MTLSize::new(inputs.len() as u64, 1, 1),
        MTLSize::new(256, 1, 1),
    );
    encoder.end_encoding();
    command.commit();
    command.wait_until_completed();
    if command.status() != MTLCommandBufferStatus::Completed {
        return Err(failure(
            "strict GPU CPU-exp compatibility diagnostic did not complete",
        ));
    }
    let gpu_fdlibm = gpu_read_f32(&fdlibm_out, inputs.len())?;
    let gpu_precise = gpu_read_f32(&precise_out, inputs.len())?;
    let gpu_sigmoid = gpu_read_f32(&sigmoid_out, inputs.len())?;

    let darwin_dd_exp_out = gpu_f32_buffer(&device, &vec![0.0; inputs.len()]);
    let darwin_dd_sigmoid_out = gpu_f32_buffer(&device, &vec![0.0; inputs.len()]);
    let darwin_dd_table = gpu_float4_buffer(&device, &darwin_expf_dd_table());
    let darwin_dd_command = queue.new_command_buffer();
    let darwin_dd_encoder = darwin_dd_command.new_compute_command_encoder();
    darwin_dd_encoder.set_compute_pipeline_state(&darwin_dd_pipeline);
    darwin_dd_encoder.set_buffer(0, Some(&input), 0);
    darwin_dd_encoder.set_buffer(1, Some(&darwin_dd_exp_out), 0);
    darwin_dd_encoder.set_buffer(2, Some(&darwin_dd_sigmoid_out), 0);
    darwin_dd_encoder.set_buffer(3, Some(&darwin_dd_table), 0);
    darwin_dd_encoder.set_bytes(
        4,
        std::mem::size_of::<u32>() as u64,
        &count as *const u32 as *const _,
    );
    darwin_dd_encoder.dispatch_threads(
        MTLSize::new(inputs.len() as u64, 1, 1),
        MTLSize::new(256, 1, 1),
    );
    darwin_dd_encoder.end_encoding();
    darwin_dd_command.commit();
    darwin_dd_command.wait_until_completed();
    if darwin_dd_command.status() != MTLCommandBufferStatus::Completed {
        return Err(failure(
            "strict GPU software-FP64 Darwin expf feasibility diagnostic did not complete",
        ));
    }
    let gpu_darwin_dd_exp = gpu_read_f32(&darwin_dd_exp_out, inputs.len())?;
    let gpu_darwin_dd_sigmoid = gpu_read_f32(&darwin_dd_sigmoid_out, inputs.len())?;
    let darwin_dd_device_cost = gpu_darwin_dd_device_cost_probe(
        &device,
        &queue,
        &darwin_dd_pipeline,
        &darwin_dd_table,
        inputs,
    )?;
    let darwin_dd_mhc_control = gpu_darwin_dd_mhc_control_probe(
        &device,
        &queue,
        &library,
        &darwin_dd_table,
        p1_mixes,
        hc_scale,
        hc_base,
        p1_post,
        p1_comb,
    )?;
    let darwin_dd_adversarial_stress = gpu_darwin_dd_adversarial_stress_probe(
        &device,
        &queue,
        &darwin_dd_pipeline,
        &darwin_dd_table,
    )?;
    if gpu_fdlibm
        .iter()
        .chain(&gpu_precise)
        .chain(&gpu_sigmoid)
        .chain(&gpu_darwin_dd_exp)
        .chain(&gpu_darwin_dd_sigmoid)
        .any(|value| !value.is_finite())
    {
        return Err(failure(
            "strict GPU CPU-exp compatibility diagnostic produced non-finite output",
        ));
    }
    const P1_CORPUS_PREFIX: usize = 6;
    let p1_end = P1_CORPUS_PREFIX
        .checked_add(p1_inputs.len())
        .ok_or_else(|| failure("P1 corpus window overflow"))?;
    if inputs.get(P1_CORPUS_PREFIX..p1_end) != Some(p1_inputs) {
        return Err(failure(
            "representative corpus no longer contains its explicit P1 window",
        ));
    }
    Ok(json!({
        "kernel":KERNEL,
        "fast_math_enabled":false,
        "fp64_language_probe":fp64_language_probe,
        "elements":inputs.len(),
        "gpu_fdlibm_vs_cpu_fdlibm":f32_delta(cpu_fdlibm_exp, &gpu_fdlibm, inputs),
        "gpu_fdlibm_vs_rust_expf":f32_delta(rust_exp, &gpu_fdlibm, inputs),
        "gpu_precise_exp_vs_rust_expf":f32_delta(rust_exp, &gpu_precise, inputs),
        "gpu_fdlibm_sigmoid_vs_rust_sigmoid":f32_delta(rust_sigmoid, &gpu_sigmoid, inputs),
        "gpu_darwin_dd_exp_vs_rust_expf":f32_delta(rust_exp, &gpu_darwin_dd_exp, inputs),
        "gpu_darwin_dd_sigmoid_vs_rust_sigmoid":f32_delta(rust_sigmoid, &gpu_darwin_dd_sigmoid, inputs),
        "gpu_darwin_dd_device_cost":darwin_dd_device_cost,
        "gpu_darwin_dd_p1_mhc_control":darwin_dd_mhc_control,
        "gpu_darwin_dd_adversarial_stress":darwin_dd_adversarial_stress,
        "p1_control_window":{
            "elements":p1_inputs.len(),
            "gpu_fdlibm_exp_vs_rust_expf":f32_delta(&rust_exp[P1_CORPUS_PREFIX..p1_end], &gpu_fdlibm[P1_CORPUS_PREFIX..p1_end], p1_inputs),
            "gpu_precise_exp_vs_rust_expf":f32_delta(&rust_exp[P1_CORPUS_PREFIX..p1_end], &gpu_precise[P1_CORPUS_PREFIX..p1_end], p1_inputs),
            "gpu_fdlibm_sigmoid_vs_rust_sigmoid":f32_delta(&rust_sigmoid[P1_CORPUS_PREFIX..p1_end], &gpu_sigmoid[P1_CORPUS_PREFIX..p1_end], p1_inputs),
            "gpu_darwin_dd_exp_vs_rust_expf":f32_delta(&rust_exp[P1_CORPUS_PREFIX..p1_end], &gpu_darwin_dd_exp[P1_CORPUS_PREFIX..p1_end], p1_inputs),
            "gpu_darwin_dd_sigmoid_vs_rust_sigmoid":f32_delta(&rust_sigmoid[P1_CORPUS_PREFIX..p1_end], &gpu_darwin_dd_sigmoid[P1_CORPUS_PREFIX..p1_end], p1_inputs),
        },
    }))
}

#[cfg(target_os = "macos")]
fn metal_fp64_language_probe(device: &Device) -> Value {
    // This compile-only probe is deliberately tiny.  It establishes whether
    // an all-device literal port of Darwin's F64 expf fast path is even a
    // valid MSL representation on the active GPU.
    const SOURCE: &str = r#"
        #include <metal_stdlib>
        using namespace metal;
        kernel void hawking_fp64_language_probe(
            device const float* input [[buffer(0)]],
            device float* output [[buffer(1)]],
            uint index [[thread_position_in_grid]]) {
            double x = double(input[index]);
            output[index] = float(x * 1.0);
        }
    "#;
    let options = CompileOptions::new();
    options.set_fast_math_enabled(false);
    match device.new_library_with_source(SOURCE, &options) {
        Ok(_) => json!({"compiled":true}),
        Err(error) => json!({"compiled":false,"diagnostic":error.to_string()}),
    }
}

#[cfg(target_os = "macos")]
fn gpu_f32_buffer(device: &Device, values: &[f32]) -> metal::Buffer {
    let buffer = device.new_buffer(
        (values.len() * std::mem::size_of::<f32>()) as u64,
        MTLResourceOptions::StorageModeShared,
    );
    unsafe {
        std::ptr::copy_nonoverlapping(
            values.as_ptr() as *const u8,
            buffer.contents() as *mut u8,
            values.len() * std::mem::size_of::<f32>(),
        );
    }
    buffer
}

#[cfg(target_os = "macos")]
fn gpu_float4_buffer(device: &Device, values: &[[f32; 4]]) -> metal::Buffer {
    let buffer = device.new_buffer(
        (values.len() * std::mem::size_of::<[f32; 4]>()) as u64,
        MTLResourceOptions::StorageModeShared,
    );
    unsafe {
        std::ptr::copy_nonoverlapping(
            values.as_ptr() as *const u8,
            buffer.contents() as *mut u8,
            values.len() * std::mem::size_of::<[f32; 4]>(),
        );
    }
    buffer
}

#[cfg(target_os = "macos")]
fn darwin_expf_dd_table() -> Vec<[f32; 4]> {
    DARWIN_EXPF_TABLE
        .iter()
        .enumerate()
        .map(|(index, &entry)| {
            let source_bits = entry.wrapping_add(
                DARWIN_EXPF_MAGIC_BITS
                    .wrapping_add(index as u64)
                    .wrapping_shl(45),
            );
            let source = f64::from_bits(source_bits);
            let high = source as f32;
            let middle = (source - f64::from(high)) as f32;
            let low = (source - f64::from(high) - f64::from(middle)) as f32;
            [high, middle, low, 0.0]
        })
        .collect()
}

#[cfg(target_os = "macos")]
fn gpu_darwin_dd_mhc_control_probe(
    device: &Device,
    queue: &metal::CommandQueue,
    library: &metal::Library,
    table: &metal::Buffer,
    mixes: &[f32],
    hc_scale: &[f32],
    hc_base: &[f32],
    expected_post: &[f32],
    expected_comb: &[f32],
) -> ProbeResult<Value> {
    const KERNEL: &str = "deepseek_v4_p4b_hc_post_comb_darwin_dd_candidate";
    if mixes.len() != 24
        || hc_scale.len() != 3
        || hc_base.len() != 24
        || expected_post.len() != 4
        || expected_comb.len() != 16
    {
        return Err(failure("P1 mHC-control feasibility geometry changed"));
    }
    let function = library.get_function(KERNEL, None).map_err(failure)?;
    let pipeline = device
        .new_compute_pipeline_state_with_function(&function)
        .map_err(failure)?;
    let mixes = gpu_f32_buffer(device, mixes);
    let hc_scale = gpu_f32_buffer(device, hc_scale);
    let hc_base = gpu_f32_buffer(device, hc_base);
    let post = gpu_f32_buffer(device, &[0.0; 4]);
    let comb = gpu_f32_buffer(device, &[0.0; 16]);
    let hc_mult = 4u32;
    let mix_width = 24u32;
    let sinkhorn_iters = 20u32;
    let command = queue.new_command_buffer();
    let encoder = command.new_compute_command_encoder();
    encoder.set_compute_pipeline_state(&pipeline);
    encoder.set_buffer(0, Some(&mixes), 0);
    encoder.set_buffer(1, Some(&hc_scale), 0);
    encoder.set_buffer(2, Some(&hc_base), 0);
    encoder.set_buffer(3, Some(&post), 0);
    encoder.set_buffer(4, Some(&comb), 0);
    encoder.set_bytes(
        5,
        std::mem::size_of::<u32>() as u64,
        &hc_mult as *const u32 as *const _,
    );
    encoder.set_bytes(
        6,
        std::mem::size_of::<u32>() as u64,
        &mix_width as *const u32 as *const _,
    );
    encoder.set_bytes(
        7,
        std::mem::size_of::<u32>() as u64,
        &sinkhorn_iters as *const u32 as *const _,
    );
    encoder.set_bytes(
        8,
        std::mem::size_of::<f32>() as u64,
        &HC_EPS as *const f32 as *const _,
    );
    encoder.set_buffer(9, Some(table), 0);
    encoder.dispatch_threads(MTLSize::new(1, 1, 1), MTLSize::new(1, 1, 1));
    encoder.end_encoding();
    command.commit();
    command.wait_until_completed();
    if command.status() != MTLCommandBufferStatus::Completed {
        return Err(failure(
            "software-FP64 P1 mHC-control feasibility dispatch did not complete",
        ));
    }
    let actual_post = gpu_read_f32(&post, 4)?;
    let actual_comb = gpu_read_f32(&comb, 16)?;
    let post_delta = f32_delta(expected_post, &actual_post, &mixes_f32_for_post(&mixes)?);
    let comb_delta = f32_delta(expected_comb, &actual_comb, &mixes_f32_for_comb(&mixes)?);
    Ok(json!({
        "kernel":KERNEL,
        "strict_math":true,
        "post_f32_bitwise_delta":post_delta,
        "comb_f32_bitwise_delta":comb_delta,
        "source_loop_order":"P4B source sigmoid + 20-iteration split Sinkhorn; only exp implementation differs",
        "dispatches":1,
        "command_buffers":1,
        "cpu_visible_waits":1,
        "fallback":false,
        "promotion":false,
    }))
}

#[cfg(target_os = "macos")]
fn mixes_f32_for_post(mixes: &metal::Buffer) -> ProbeResult<Vec<f32>> {
    let source = gpu_read_f32(mixes, 24)?;
    Ok(source[4..8].to_vec())
}

#[cfg(target_os = "macos")]
fn mixes_f32_for_comb(mixes: &metal::Buffer) -> ProbeResult<Vec<f32>> {
    let source = gpu_read_f32(mixes, 24)?;
    Ok(source[8..24].to_vec())
}

#[cfg(target_os = "macos")]
fn gpu_darwin_dd_adversarial_stress_probe(
    device: &Device,
    queue: &metal::CommandQueue,
    pipeline: &metal::ComputePipelineState,
    table: &metal::Buffer,
) -> ProbeResult<Value> {
    let inputs = darwin_dd_adversarial_inputs();
    let expected_exp: Vec<f32> = inputs.iter().map(|&value| value.exp()).collect();
    let expected_sigmoid: Vec<f32> = inputs.iter().map(|&value| sigmoid_rust(value)).collect();
    let count = u32::try_from(inputs.len())
        .map_err(|_| failure("software-FP64 adversarial corpus exceeds Metal u32 grid"))?;
    let input = gpu_f32_buffer(device, &inputs);
    let exp_out = gpu_f32_buffer(device, &vec![0.0; inputs.len()]);
    let sigmoid_out = gpu_f32_buffer(device, &vec![0.0; inputs.len()]);
    let command = queue.new_command_buffer();
    let encoder = command.new_compute_command_encoder();
    encoder.set_compute_pipeline_state(pipeline);
    encoder.set_buffer(0, Some(&input), 0);
    encoder.set_buffer(1, Some(&exp_out), 0);
    encoder.set_buffer(2, Some(&sigmoid_out), 0);
    encoder.set_buffer(3, Some(table), 0);
    encoder.set_bytes(
        4,
        std::mem::size_of::<u32>() as u64,
        &count as *const u32 as *const _,
    );
    encoder.dispatch_threads(
        MTLSize::new(inputs.len() as u64, 1, 1),
        MTLSize::new(256, 1, 1),
    );
    encoder.end_encoding();
    command.commit();
    command.wait_until_completed();
    if command.status() != MTLCommandBufferStatus::Completed {
        return Err(failure(
            "software-FP64 adversarial-domain feasibility dispatch did not complete",
        ));
    }
    let actual_exp = gpu_read_f32(&exp_out, inputs.len())?;
    let actual_sigmoid = gpu_read_f32(&sigmoid_out, inputs.len())?;
    let (_, _, duration_ns) = gpu_timestamp_ns(&command)?;
    Ok(json!({
        "status":"GENERAL_DOMAIN_STRESS_DIAGNOSTIC_ONLY_NOT_A_PROOF_OVER_ALL_F32",
        "inputs":inputs.len(),
        "construction":{
            "normal_and_subnormal_bit_patterns":1_048_576,
            "range_reduction_boundary_neighbors":"nearest F32 values on both sides of every n+0.5 128/ln2 table-selection boundary in [-40,40]",
            "special_values":"signed zero, signed least subnormal, signed least normal, signed one, signed 40",
        },
        "gpu_exp_vs_rust_expf":f32_delta(&expected_exp, &actual_exp, &inputs),
        "gpu_sigmoid_vs_rust_sigmoid":f32_delta(&expected_sigmoid, &actual_sigmoid, &inputs),
        "gpu_duration_ns":duration_ns,
        "dispatches":1,
        "command_buffers":1,
        "fallback":false,
    }))
}

fn darwin_dd_adversarial_inputs() -> Vec<f32> {
    const RANDOM_NORMAL_OR_SUBNORMAL_INPUTS: usize = 1_048_576;
    let mut values = vec![
        -40.0,
        40.0,
        -1.0,
        1.0,
        -0.0,
        0.0,
        f32::from_bits(1),
        f32::from_bits(0x8000_0001),
        f32::from_bits(0x0080_0000),
        f32::from_bits(0x8080_0000),
    ];
    let inv_ln2_x128 = f64::from_bits(DARWIN_EXPF_INV_LN2_X128_BITS);
    for n in -7_388i32..=7_388 {
        let threshold = (f64::from(n) + 0.5) / inv_ln2_x128;
        let rounded = threshold as f32;
        if rounded.is_finite() && (DOMAIN_MIN..=DOMAIN_MAX).contains(&rounded) {
            values.push(next_down(rounded));
            values.push(rounded);
            values.push(next_up(rounded));
        }
    }
    let mut state = 0x84d9_1b0d_aec5_413fu64;
    let target = values
        .len()
        .checked_add(RANDOM_NORMAL_OR_SUBNORMAL_INPUTS)
        .expect("adversarial control corpus length fits usize");
    while values.len() < target {
        state = state
            .wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(1);
        let sign = ((state >> 63) as u32) << 31;
        let exponent = ((state >> 32) as u32) % 133;
        let mantissa = (state as u32) & 0x007f_ffff;
        let value = f32::from_bits(sign | (exponent << 23) | mantissa);
        if value.is_finite() && (DOMAIN_MIN..=DOMAIN_MAX).contains(&value) {
            values.push(value);
        }
    }
    values
}

#[cfg(target_os = "macos")]
fn gpu_darwin_dd_device_cost_probe(
    device: &Device,
    queue: &metal::CommandQueue,
    pipeline: &metal::ComputePipelineState,
    table: &metal::Buffer,
    corpus: &[f32],
) -> ProbeResult<Value> {
    const CORPUS_REPETITIONS: usize = 64;
    const WARMUPS: usize = 2;
    const TRIALS: usize = 5;
    let total_inputs = corpus
        .len()
        .checked_mul(CORPUS_REPETITIONS)
        .ok_or_else(|| failure("software-FP64 timing corpus length overflow"))?;
    let count = u32::try_from(total_inputs)
        .map_err(|_| failure("software-FP64 timing corpus exceeds Metal u32 grid"))?;
    let mut repeated_inputs = Vec::with_capacity(total_inputs);
    for _ in 0..CORPUS_REPETITIONS {
        repeated_inputs.extend_from_slice(corpus);
    }
    let input = gpu_f32_buffer(device, &repeated_inputs);
    let exp_out = gpu_f32_buffer(device, &vec![0.0; total_inputs]);
    let sigmoid_out = gpu_f32_buffer(device, &vec![0.0; total_inputs]);
    let mut durations_ns = Vec::with_capacity(TRIALS);
    for trial in 0..(WARMUPS + TRIALS) {
        let command = queue.new_command_buffer();
        let encoder = command.new_compute_command_encoder();
        encoder.set_compute_pipeline_state(pipeline);
        encoder.set_buffer(0, Some(&input), 0);
        encoder.set_buffer(1, Some(&exp_out), 0);
        encoder.set_buffer(2, Some(&sigmoid_out), 0);
        encoder.set_buffer(3, Some(table), 0);
        encoder.set_bytes(
            4,
            std::mem::size_of::<u32>() as u64,
            &count as *const u32 as *const _,
        );
        encoder.dispatch_threads(
            MTLSize::new(total_inputs as u64, 1, 1),
            MTLSize::new(256, 1, 1),
        );
        encoder.end_encoding();
        command.commit();
        command.wait_until_completed();
        if command.status() != MTLCommandBufferStatus::Completed {
            return Err(failure(format!(
                "software-FP64 timing trial {trial} did not complete"
            )));
        }
        if trial >= WARMUPS {
            let (_, _, duration_ns) = gpu_timestamp_ns(&command)?;
            durations_ns.push(duration_ns);
        }
    }
    let mut sorted = durations_ns.clone();
    sorted.sort_unstable();
    let percentile = |numerator: usize, denominator: usize| -> u64 {
        let index = ((sorted.len() - 1) * numerator + (denominator / 2)) / denominator;
        sorted[index]
    };
    let exp_calls_per_dispatch = (total_inputs as u64)
        .checked_mul(2)
        .ok_or_else(|| failure("software-FP64 exp call count overflow"))?;
    let p50_ns = percentile(50, 100);
    let p95_ns = percentile(95, 100);
    let p99_ns = percentile(99, 100);
    Ok(json!({
        "status":"GPU_TIMESTAMPED_DIAGNOSTIC_ONLY_NOT_A_RUNTIME_BENCHMARK",
        "kernel":"deepseek_v4_p4b_darwin_expf_dd_compat_domain_candidate",
        "warmup_dispatches":WARMUPS,
        "clean_timestamped_dispatches":TRIALS,
        "inputs_per_dispatch":total_inputs,
        "darwin_expf_evaluations_per_input":2,
        "darwin_expf_evaluations_per_dispatch":exp_calls_per_dispatch,
        "gpu_duration_ns_samples":durations_ns,
        "gpu_duration_ns":{"p50":p50_ns,"p95":p95_ns,"p99":p99_ns},
        "end_to_end_kernel_ns_per_darwin_expf":{"p50":p50_ns as f64 / exp_calls_per_dispatch as f64,"p95":p95_ns as f64 / exp_calls_per_dispatch as f64,"p99":p99_ns as f64 / exp_calls_per_dispatch as f64},
        "logical_input_bytes_per_dispatch":(total_inputs * std::mem::size_of::<f32>()) as u64,
        "logical_output_bytes_per_dispatch":(total_inputs * 2 * std::mem::size_of::<f32>()) as u64,
        "table_constant_bytes":(128 * std::mem::size_of::<[f32;4]>()) as u64,
        "threadgroup_size":256,
        "command_buffers_per_dispatch":1,
        "compute_dispatches_per_command_buffer":1,
        "cpu_visible_waits_per_dispatch":1,
        "fallback":false,
        "timing_authority":"completed MTLCommandBuffer GPUStartTime/GPUEndTime; host allocation/encode/submit/wait are excluded",
        "cost_scope":"two scalar Darwin-exp evaluations plus two global F32 output writes per thread; this is not full mHC, token, or TPS cost",
    }))
}

#[cfg(target_os = "macos")]
fn gpu_timestamp_ns(command: &metal::CommandBufferRef) -> ProbeResult<(u64, u64, u64)> {
    let (start, end): (f64, f64) = unsafe {
        (
            msg_send![command, GPUStartTime],
            msg_send![command, GPUEndTime],
        )
    };
    if !(start.is_finite() && end.is_finite() && start > 0.0 && end > start) {
        return Err(failure(format!(
            "completed software-FP64 command lacks valid GPU timestamps: start={start:?} end={end:?}"
        )));
    }
    let duration_ns = ((end - start) * 1_000_000_000.0).round() as u64;
    if duration_ns == 0 {
        return Err(failure("software-FP64 GPU duration rounded to zero"));
    }
    Ok((
        (start * 1_000_000_000.0).round() as u64,
        (end * 1_000_000_000.0).round() as u64,
        duration_ns,
    ))
}

#[cfg(target_os = "macos")]
fn gpu_read_f32(buffer: &metal::Buffer, count: usize) -> ProbeResult<Vec<f32>> {
    if buffer.length() < (count * std::mem::size_of::<f32>()) as u64 {
        return Err(failure("GPU CPU-exp diagnostic output read overflow"));
    }
    Ok(unsafe { std::slice::from_raw_parts(buffer.contents() as *const f32, count).to_vec() })
}

fn f32_delta(expected: &[f32], actual: &[f32], inputs: &[f32]) -> Value {
    assert_eq!(expected.len(), actual.len());
    assert_eq!(expected.len(), inputs.len());
    let mut mismatch_count = 0usize;
    let mut max_ulp_delta = 0u32;
    let mut first_mismatch = None;
    for (index, ((&expected, &actual), &input)) in
        expected.iter().zip(actual).zip(inputs).enumerate()
    {
        if expected.to_bits() != actual.to_bits() {
            mismatch_count += 1;
            max_ulp_delta = max_ulp_delta.max(expected.to_bits().abs_diff(actual.to_bits()));
            if first_mismatch.is_none() {
                first_mismatch = Some(json!({
                    "index":index,
                    "input_bits":format!("0x{:08x}",input.to_bits()),
                    "input_value":input.to_string(),
                    "expected_bits":format!("0x{:08x}",expected.to_bits()),
                    "actual_bits":format!("0x{:08x}",actual.to_bits()),
                    "expected_value":expected.to_string(),
                    "actual_value":actual.to_string(),
                }));
            }
        }
    }
    json!({"elements":expected.len(),"bitwise_mismatch_count":mismatch_count,"max_raw_word_delta":max_ulp_delta,"first_mismatch":first_mismatch})
}

fn f32_words(values: &[f32]) -> Vec<String> {
    values
        .iter()
        .map(|value| format!("0x{:08x}", value.to_bits()))
        .collect()
}

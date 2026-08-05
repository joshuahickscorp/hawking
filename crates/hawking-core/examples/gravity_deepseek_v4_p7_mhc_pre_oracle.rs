//! Raw-artifact CPU reference for the isolated DeepSeek-V4 P7 mHC-FFN pre
//! preparation, plus a compile-only smoke of its standalone Metal source.
//!
//! This is intentionally neither a receipt producer nor a runtime path.  It
//! reads one bounded layer-0/position-1 source trace, computes only
//! `P4B-attention state -> hc_ffn_pre -> ffn RMSNorm`, and compiles the three
//! unregistered P7 kernels without allocating a model runtime or dispatching
//! GPU work.

use half::bf16;
use hawking_core::gravity_deepseek_v4::DeepSeekV4FullStreamReader;
use hawking_core::gravity_deepseek_v4_layer0_attention::rms_norm_bf16_source_algorithm;
use hawking_core::gravity_deepseek_v4_layer0_continuation::{
    layer0_position1_complete_attention_cpu_oracle, verify_layer0_position1_continuation_anchors,
    POSITION1, POSITION1_TOKEN_ID,
};
use hawking_core::gravity_deepseek_v4_layer0_moe::{
    LAYER0_FFN_NORM_WEIGHT, LAYER0_HC_FFN_BASE, LAYER0_HC_FFN_FN, LAYER0_HC_FFN_SCALE,
};
use hawking_core::gravity_deepseek_v4_layer0_prefix::{
    hc_attn_pre_source_algorithm, HC_EPS, HC_FLAT_WIDTH, HC_MIX_WIDTH, HC_SINKHORN_ITERS,
    HIDDEN_SIZE, RMS_NORM_EPS,
};
use serde_json::json;
use sha2::{Digest, Sha256};
use std::error::Error;
use std::path::PathBuf;

const STATUS: &str = "P7_RAW_ARTIFACT_CPU_PRE_ORACLE_SHADER_COMPILE_ONLY_NOT_RUNTIME";
const P7_KERNELS: &[&str] = &[
    "deepseek_v4_p7_mhc_ffn_pre_authority",
    "deepseek_v4_p7_ffn_rmsnorm_bf16_authority",
    "deepseek_v4_p7_mhc_ffn_post_authority",
];

type ExampleResult<T> = Result<T, Box<dyn Error>>;

struct Args {
    artifact: PathBuf,
}

fn main() -> ExampleResult<()> {
    let args = parse_args()?;
    let reader = DeepSeekV4FullStreamReader::admit(&args.artifact)?;
    verify_layer0_position1_continuation_anchors(&reader)?;
    let attention = layer0_position1_complete_attention_cpu_oracle(&reader)?;
    if attention.causal.token1_id != POSITION1_TOKEN_ID
        || attention.causal.token1_prefix.token_id != POSITION1_TOKEN_ID
        || attention.causal.kv_cache_two_rows_bf16_bits.len() != 2 * 512
        || attention.hc_attention_post_bf16_bits.len() != HC_FLAT_WIDTH
    {
        return Err("P7 raw-artifact oracle received an invalid P4B CPU predecessor".into());
    }

    let hc_fn = read_f32_tensor(&reader, LAYER0_HC_FFN_FN, HC_MIX_WIDTH * HC_FLAT_WIDTH)?;
    let hc_base = read_f32_tensor(&reader, LAYER0_HC_FFN_BASE, HC_MIX_WIDTH)?;
    let hc_scale = read_f32_tensor(&reader, LAYER0_HC_FFN_SCALE, 3)?;
    let ffn_weight = read_bf16_tensor(&reader, LAYER0_FFN_NORM_WEIGHT, HIDDEN_SIZE)?;
    let (flat_rsqrt, mixes, pre, post, comb, reduced) = hc_attn_pre_source_algorithm(
        &attention.hc_attention_post_bf16_bits,
        &hc_fn,
        &hc_scale,
        &hc_base,
        RMS_NORM_EPS,
        HC_EPS,
        HC_SINKHORN_ITERS,
    )?;
    let ffn_norm =
        rms_norm_bf16_source_algorithm(&reduced, &ffn_weight, HIDDEN_SIZE, RMS_NORM_EPS)?;
    if !flat_rsqrt.is_finite()
        || mixes
            .iter()
            .chain(&pre)
            .chain(&post)
            .chain(&comb)
            .any(|v| !v.is_finite())
        || reduced.len() != HIDDEN_SIZE
        || ffn_norm.len() != HIDDEN_SIZE
    {
        return Err("P7 raw-artifact CPU pre-oracle produced invalid output".into());
    }

    let shader_device = compile_p7_through_metal_context()?;
    println!(
        "{}",
        serde_json::to_string_pretty(&json!({
            "status": STATUS,
            "scope": {
                "layer": 0,
                "token_id": POSITION1_TOKEN_ID,
                "token_position": POSITION1,
                "batch": 1,
                "p4b_cpu_predecessor": "complete attention state BF16[4,4096] plus causal KV BF16[2,512]",
                "executed_cpu_operator_chain": "mHC-FFN pre/Sinkhorn -> FFn RMSNorm",
                "p6_executed": false,
                "mHC-FFN post_executed": false,
            },
            "artifact": {
                "path": reader.artifact_root(),
                "manifest_seal_sha256": reader.manifest_seal_sha256(),
                "source_revision": reader.source_identity().revision,
                "source_parent_retained": false,
            },
            "verified_source_controls": [
                tensor_summary(&reader, LAYER0_HC_FFN_FN)?,
                tensor_summary(&reader, LAYER0_HC_FFN_BASE)?,
                tensor_summary(&reader, LAYER0_HC_FFN_SCALE)?,
                tensor_summary(&reader, LAYER0_FFN_NORM_WEIGHT)?,
            ],
            "bounded_cpu_checkpoints": {
                "attention_hc_post_bf16_sha256": sha256_u16(&attention.hc_attention_post_bf16_bits),
                "causal_kv_two_rows_bf16_sha256": sha256_u16(&attention.causal.kv_cache_two_rows_bf16_bits),
                "ffn_pre_reduced_bf16_sha256": sha256_u16(&reduced),
                "ffn_norm_bf16_sha256": sha256_u16(&ffn_norm),
                "ffn_post_f32_sha256": sha256_f32(&post),
                "ffn_comb_f32_sha256": sha256_f32(&comb),
            },
            "future_p4b_device_join_reporting": {
                "exact_storage": "NOT_MEASURED_BY_CPU_ORACLE",
                "numeric_parity_v2_1": "NOT_MEASURED_BY_CPU_ORACLE",
                "join_as_exact_predecessor": false,
                "policy": "A future joined P7 run must report exact-storage and Numeric Parity V2.1 separately; V2.1-only state cannot be described as an exact P7 continuation.",
            },
            "metal_context_library_compile": {
                "metal_device": shader_device,
                "kernels": P7_KERNELS,
                "dispatches": 0,
                "command_buffers": 0,
                "cpu_visible_waits": 0,
            },
            "claim_boundary": "This is a bounded raw-artifact CPU source-algorithm reference and isolated shader compile only. It does not establish P4B device parity, a joined P4B/P6 device path, a P6/MoE result, mHC-FFN post result, runtime, Engine, HCLI, generation, or TPS."
        }))?
    );
    Ok(())
}

fn parse_args() -> ExampleResult<Args> {
    let mut artifact = None;
    let mut args = std::env::args_os().skip(1);
    while let Some(flag) = args.next() {
        match flag.to_string_lossy().as_ref() {
            "--artifact" => artifact = args.next().map(PathBuf::from),
            other => return Err(format!("unknown argument {other}").into()),
        }
    }
    Ok(Args {
        artifact: artifact.ok_or("--artifact required")?,
    })
}

fn read_f32_tensor(
    reader: &DeepSeekV4FullStreamReader,
    name: &str,
    expected_elements: usize,
) -> ExampleResult<Vec<f32>> {
    let metadata = reader.tensor_metadata(name)?;
    if metadata.dtype != "F32" || metadata.bytes != (expected_elements * 4) as u64 {
        return Err(format!("{name} does not match expected F32 source geometry").into());
    }
    let bytes = reader.read_verified_full(name, expected_elements * 4)?;
    let values = bytes
        .chunks_exact(4)
        .map(|chunk| f32::from_bits(u32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]])))
        .collect::<Vec<_>>();
    if values.len() != expected_elements || values.iter().any(|value| !value.is_finite()) {
        return Err(format!("{name} contains invalid F32 source data").into());
    }
    Ok(values)
}

fn read_bf16_tensor(
    reader: &DeepSeekV4FullStreamReader,
    name: &str,
    expected_elements: usize,
) -> ExampleResult<Vec<u16>> {
    let metadata = reader.tensor_metadata(name)?;
    if metadata.dtype != "BF16" || metadata.bytes != (expected_elements * 2) as u64 {
        return Err(format!("{name} does not match expected BF16 source geometry").into());
    }
    let bytes = reader.read_verified_full(name, expected_elements * 2)?;
    let values = bytes
        .chunks_exact(2)
        .map(|chunk| u16::from_le_bytes([chunk[0], chunk[1]]))
        .collect::<Vec<_>>();
    if values.len() != expected_elements
        || values
            .iter()
            .any(|bits| !bf16::from_bits(*bits).to_f32().is_finite())
    {
        return Err(format!("{name} contains invalid BF16 source data").into());
    }
    Ok(values)
}

fn tensor_summary(
    reader: &DeepSeekV4FullStreamReader,
    name: &str,
) -> ExampleResult<serde_json::Value> {
    let tensor = reader.tensor_metadata(name)?;
    Ok(json!({
        "name": tensor.name,
        "dtype": tensor.dtype,
        "shape": tensor.shape,
        "bytes": tensor.bytes,
        "verified_before_cpu_use": true,
        "source_shard": tensor.source_shard,
    }))
}

fn sha256_u16(values: &[u16]) -> String {
    let mut hash = Sha256::new();
    for value in values {
        hash.update(value.to_le_bytes());
    }
    format!("{:x}", hash.finalize())
}

fn sha256_f32(values: &[f32]) -> String {
    let mut hash = Sha256::new();
    for value in values {
        hash.update(value.to_bits().to_le_bytes());
    }
    format!("{:x}", hash.finalize())
}

#[cfg(target_os = "macos")]
fn compile_p7_through_metal_context() -> ExampleResult<String> {
    use hawking_core::metal::MetalContext;

    let context = MetalContext::new()?;
    for &kernel in P7_KERNELS {
        context
            .pipeline(kernel)
            .map_err(|error| format!("P7 MetalContext pipeline {kernel} failed: {error}"))?;
    }
    Ok(context.device_name())
}

#[cfg(not(target_os = "macos"))]
fn compile_p7_through_metal_context() -> ExampleResult<String> {
    Err("P7 MetalContext compile smoke requires macOS".into())
}

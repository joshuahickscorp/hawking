//! Native Rust F32 formula control for the pinned Flash PLE BOS fixture.
//!
//! This consumes an existing independently checked source-bound fixture. It
//! range-reads exact BF16 source rows/support, evaluates PLE on CPU in Rust,
//! and compares injection plus persistent convolution state. It is not the
//! final Metal path and cannot authorize a source teacher or runtime.

use hawking_core::flash_ple::{
    FlashPleAddressConfig, FlashPleAddressContract, FlashPleBf16Support, FlashPleLookupLayout,
};
use hawking_core::gravity_deepseek_v4::canonical_json;
use hawking_core::model::qwen80_source_bf16_layer_major::SourceBf16Index;
use serde_json::{json, Value};
use sha2::{Digest, Sha256};
use std::error::Error;
use std::fs::{self, OpenOptions};
use std::io::Write;
use std::path::{Path, PathBuf};
use std::time::{SystemTime, UNIX_EPOCH};

const SCHEMA: &str = "hawking.flash.ple_native_formula_control.v1";
const PASS: &str = "NATIVE_RUST_PLE_FORMULA_BOUNDED_PARITY_PASS";
const FAIL: &str = "NATIVE_RUST_PLE_FORMULA_BOUNDED_PARITY_FAIL";
const EXPECTED_CONFIG_SHA256: &str =
    "889658f2508e8c61d409b02e70e0d78d8d4452ec65aaafbe129805d213d2e74b";
const EXPECTED_INDEX_SHA256: &str =
    "99e815241ef03325536b0aaa4441deea45174c17fae31e10f0bb456410c590de";
const EXPECTED_INPUT_SHA256: &str =
    "459e0084491336f1b4a1b4b8af02c65f0603d2fb1c46009d0ad878ef0b9bbc3e";
const EXPECTED_INJECTION_SHA256: &str =
    "2761917fb36a92fd579689800a0e9eb10d67de90c57811df0a3b12ff4f7721d8";
const EXPECTED_STATE_SHA256: &str =
    "76def3e11f235c0285e681d6d2b4f5c080b5fe69228ccfa4f0ebf079d306cc36";
const CONTROL_TOKEN_ID: u64 = 248_044;
const HIDDEN: usize = 2_560;
const HC_COUNT: usize = 4;
const HC: usize = HIDDEN * HC_COUNT;
const CONV_KERNEL: usize = 4;
const NGRAM_SIZE: usize = 3;
const EPS: f32 = 1.0e-6;
const MAX_ABS_THRESHOLD: f64 = 5.0e-6;
const RELATIVE_L2_THRESHOLD: f64 = 5.0e-6;

#[derive(Debug)]
struct Args {
    root: PathBuf,
    input_state: PathBuf,
    expected_injection: PathBuf,
    expected_state: PathBuf,
    out: PathBuf,
}

fn parse_args() -> Result<Args, Box<dyn Error>> {
    let mut root = None;
    let mut input_state = None;
    let mut expected_injection = None;
    let mut expected_state = None;
    let mut out = None;
    let mut argv = std::env::args().skip(1);
    while let Some(flag) = argv.next() {
        let value = argv
            .next()
            .ok_or_else(|| format!("missing value for {flag}"))?;
        match flag.as_str() {
            "--root" => root = Some(value),
            "--input-state" => input_state = Some(value),
            "--expected-injection" => expected_injection = Some(value),
            "--expected-state" => expected_state = Some(value),
            "--out" => out = Some(value),
            other => return Err(format!("unknown argument: {other}").into()),
        }
    }
    fn path(value: Option<String>, flag: &str) -> Result<PathBuf, Box<dyn Error>> {
        value
            .map(PathBuf::from)
            .ok_or_else(|| format!("missing required {flag}").into())
    }
    Ok(Args {
        root: path(root, "--root")?,
        input_state: path(input_state, "--input-state")?,
        expected_injection: path(expected_injection, "--expected-injection")?,
        expected_state: path(expected_state, "--expected-state")?,
        out: path(out, "--out")?,
    })
}

fn sha256_bytes(bytes: &[u8]) -> String {
    let mut digest = Sha256::new();
    digest.update(bytes);
    format!("{:x}", digest.finalize())
}

fn read_exact(
    path: &Path,
    expected_sha: &str,
    expected_bytes: usize,
) -> Result<Vec<u8>, Box<dyn Error>> {
    if path.is_symlink() || !path.is_file() {
        return Err(format!("control input is absent or symlinked: {}", path.display()).into());
    }
    let bytes = fs::read(path)?;
    let sha = sha256_bytes(&bytes);
    if bytes.len() != expected_bytes || sha != expected_sha {
        return Err(format!(
            "control input differs: {} bytes={} sha256={sha}",
            path.display(),
            bytes.len()
        )
        .into());
    }
    Ok(bytes)
}

fn read_identity(path: &Path, expected_sha: &str) -> Result<Vec<u8>, Box<dyn Error>> {
    if path.is_symlink() || !path.is_file() {
        return Err(format!("source identity is absent or symlinked: {}", path.display()).into());
    }
    let bytes = fs::read(path)?;
    let sha = sha256_bytes(&bytes);
    if sha != expected_sha {
        return Err(format!("source identity differs: {} sha256={sha}", path.display()).into());
    }
    Ok(bytes)
}

fn f32_values(bytes: &[u8]) -> Vec<f32> {
    bytes
        .chunks_exact(4)
        .map(|chunk| f32::from_le_bytes(chunk.try_into().expect("four-byte chunk")))
        .collect()
}

fn f32_bytes(values: &[f32]) -> Vec<u8> {
    values
        .iter()
        .flat_map(|value| value.to_le_bytes())
        .collect()
}

fn bf16_values(bytes: &[u8]) -> Vec<f32> {
    bytes
        .chunks_exact(2)
        .map(|chunk| {
            f32::from_bits(
                (u16::from_le_bytes(chunk.try_into().expect("two-byte chunk")) as u32) << 16,
            )
        })
        .collect()
}

fn metrics(reference: &[f32], candidate: &[f32]) -> Result<Value, Box<dyn Error>> {
    if reference.is_empty() || reference.len() != candidate.len() {
        return Err("numerical control vectors have incompatible extent".into());
    }
    let mut max_abs = 0.0f64;
    let mut error_sq = 0.0f64;
    let mut reference_sq = 0.0f64;
    for (&expected, &observed) in reference.iter().zip(candidate) {
        if !expected.is_finite() || !observed.is_finite() {
            return Err("numerical control vectors contain non-finite values".into());
        }
        let error = f64::from(observed) - f64::from(expected);
        max_abs = max_abs.max(error.abs());
        error_sq += error * error;
        reference_sq += f64::from(expected) * f64::from(expected);
    }
    let relative_l2 = if reference_sq == 0.0 {
        if error_sq == 0.0 {
            0.0
        } else {
            f64::INFINITY
        }
    } else {
        (error_sq / reference_sq).sqrt()
    };
    Ok(json!({
        "finite": true,
        "elements": reference.len(),
        "max_abs": max_abs,
        "relative_l2": relative_l2,
        "within_predeclared_bound": max_abs <= MAX_ABS_THRESHOLD && relative_l2 <= RELATIVE_L2_THRESHOLD,
    }))
}

fn derived_path(out: &Path, suffix: &str) -> Result<PathBuf, Box<dyn Error>> {
    let stem = out
        .file_stem()
        .and_then(|value| value.to_str())
        .ok_or("output has no UTF-8 stem")?;
    Ok(out.with_file_name(format!("{stem}.{suffix}")))
}

fn write_new(path: &Path, bytes: &[u8]) -> Result<(), Box<dyn Error>> {
    if path.is_symlink() {
        return Err(format!("output may not be a symlink: {}", path.display()).into());
    }
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent)?;
    }
    let mut file = OpenOptions::new().create_new(true).write(true).open(path)?;
    file.write_all(bytes)?;
    file.sync_all()?;
    Ok(())
}

fn file_identity(path: &Path, bytes: &[u8]) -> Value {
    json!({"path": path, "sha256": sha256_bytes(bytes), "bytes": bytes.len()})
}

fn seal(document: &mut Value) -> Result<String, Box<dyn Error>> {
    if document.get("seal_sha256").is_some() {
        return Err("receipt already has a seal".into());
    }
    let digest = sha256_bytes(&canonical_json(document));
    document["seal_sha256"] = Value::String(digest.clone());
    Ok(digest)
}

fn support_bindings(support: &FlashPleBf16Support) -> Value {
    json!({
        "key_proj": {"bytes": support.key_proj.len(), "sha256": sha256_bytes(&support.key_proj)},
        "value_proj": {"bytes": support.value_proj.len(), "sha256": sha256_bytes(&support.value_proj)},
        "norm_key": {"bytes": support.norm_key.len(), "sha256": sha256_bytes(&support.norm_key)},
        "norm_query": {"bytes": support.norm_query.len(), "sha256": sha256_bytes(&support.norm_query)},
        "norm_conv": {"bytes": support.norm_conv.len(), "sha256": sha256_bytes(&support.norm_conv)},
        "conv1d": {"bytes": support.conv.len(), "sha256": sha256_bytes(&support.conv)},
    })
}

fn run(args: Args) -> Result<(String, String), Box<dyn Error>> {
    let injection_out = derived_path(&args.out, "native_injection.f32")?;
    let state_out = derived_path(&args.out, "native_conv_state.f32")?;
    for path in [&args.out, &injection_out, &state_out] {
        if path.exists() || path.is_symlink() {
            return Err(format!("refusing existing output: {}", path.display()).into());
        }
    }
    let started = SystemTime::now().duration_since(UNIX_EPOCH)?.as_nanos();
    let root = fs::canonicalize(&args.root)?;
    let config_path = root.join("config.json");
    let index_path = root.join("model.safetensors.index.json");
    let config = read_identity(&config_path, EXPECTED_CONFIG_SHA256)?;
    let index_bytes = read_identity(&index_path, EXPECTED_INDEX_SHA256)?;
    let input_bytes = read_exact(&args.input_state, EXPECTED_INPUT_SHA256, HC * 4)?;
    let expected_injection_bytes =
        read_exact(&args.expected_injection, EXPECTED_INJECTION_SHA256, HC * 4)?;
    let expected_state_bytes = read_exact(
        &args.expected_state,
        EXPECTED_STATE_SHA256,
        (CONV_KERNEL - 1) * NGRAM_SIZE * HC * 4,
    )?;

    let source = SourceBf16Index::open(&root)?;
    let contract =
        FlashPleAddressContract::from_config(FlashPleAddressConfig::qwen38_flash_next_layer1())?;
    let layout = FlashPleLookupLayout::from_source_index(&source, &contract, 128)?;
    let mut address_state = contract.new_state();
    let token_access = contract.trace_segment(&mut address_state, &[CONTROL_TOKEN_ID])?;
    let rows = layout.read_token_embedding_bf16(&source, &contract, &token_access[0])?;
    let embedding = rows
        .iter()
        .flat_map(|row| bf16_values(&row.bf16_bytes))
        .collect::<Vec<_>>();
    if embedding.len() != contract.config.ple_embed_dim {
        return Err("source rows did not form one complete PLE embedding".into());
    }
    let row_bindings = rows
        .iter()
        .map(|row| {
            json!({
                "global_head": row.global_head,
                "combined_row": row.combined_row,
                "shard_ordinal": row.shard_ordinal,
                "local_row": row.local_row,
                "tensor": row.tensor_name,
                "payload_sha256": sha256_bytes(&row.bf16_bytes),
                "bytes": row.bf16_bytes.len(),
            })
        })
        .collect::<Vec<_>>();
    let support =
        FlashPleBf16Support::load_source(&source, &contract, HIDDEN, HC_COUNT, CONV_KERNEL)?;
    let support_identity = support_bindings(&support);
    let result = support.evaluate_f32(
        &f32_values(&input_bytes),
        &embedding,
        1,
        NGRAM_SIZE,
        EPS,
        support.zero_state(NGRAM_SIZE)?,
    )?;
    let injection_bytes = f32_bytes(&result.injection);
    let state_bytes = f32_bytes(&result.state.conv_state);
    let injection_metrics = metrics(&f32_values(&expected_injection_bytes), &result.injection)?;
    let state_metrics = metrics(&f32_values(&expected_state_bytes), &result.state.conv_state)?;
    let passed = injection_metrics["within_predeclared_bound"] == Value::Bool(true)
        && state_metrics["within_predeclared_bound"] == Value::Bool(true);
    write_new(&injection_out, &injection_bytes)?;
    write_new(&state_out, &state_bytes)?;
    let finished = SystemTime::now().duration_since(UNIX_EPOCH)?.as_nanos();
    let status = if passed { PASS } else { FAIL };
    let mut receipt = json!({
        "schema": SCHEMA,
        "status": status,
        "started_unix_ns": started,
        "finished_unix_ns": finished,
        "elapsed_ns": finished - started,
        "source": {
            "root": root,
            "model": "Qwen/Qwen3.8-Flash-Next",
            "pinned_revision": "34567a4712bc9766c4449e2e98e4468bfa24d915",
            "config": file_identity(&config_path, &config),
            "safetensors_index": file_identity(&index_path, &index_bytes),
            "payload_bytes_read": source.bytes_read_total(),
        },
        "control": {
            "token_id": CONTROL_TOKEN_ID,
            "input_state": file_identity(&args.input_state, &input_bytes),
            "expected_injection": file_identity(&args.expected_injection, &expected_injection_bytes),
            "expected_conv_state": file_identity(&args.expected_state, &expected_state_bytes),
            "independent_reference_receipt": "receipts/headless/FLASH_PLE_LAYER1_BOS_REFERENCE_FORMULA_PARITY.json",
            "independent_reference_status": "REFERENCE_FRAMEWORK_PLE_FORMULA_PARITY_PASS",
        },
        "source_rows": row_bindings,
        "source_support": support_identity,
        "numerical_policy": {
            "support_dtype": "BF16 decoded to F32",
            "activation_and_accumulation_dtype": "F32",
            "max_abs_threshold": MAX_ABS_THRESHOLD,
            "relative_l2_threshold": RELATIVE_L2_THRESHOLD,
            "convolution": "four-tap depthwise causal cross-correlation; dilation=3; zero initial state",
        },
        "observed": {
            "injection": file_identity(&injection_out, &injection_bytes),
            "injection_metrics": injection_metrics,
            "conv_state": file_identity(&state_out, &state_bytes),
            "conv_state_metrics": state_metrics,
        },
        "model_loaded": false,
        "gpu_or_metal_started": false,
        "source_lookup_table_materialized": false,
        "complete_ple_runtime": false,
        "promotion_allowed": false,
        "claim_boundary": "One pinned source-bound BOS PLE formula fixture evaluated in native Rust CPU F32. Passing proves only this address/row/support/formula/state control. It does not establish the current owner-gated teacher trajectory, complete native PLE integration, Metal performance, full inference, compact PLE/NR, EBPW/TPS/capability, deployment, Pulsar promotion, or Kimi retirement.",
    });
    let receipt_seal = seal(&mut receipt)?;
    let mut receipt_bytes = serde_json::to_vec_pretty(&receipt)?;
    receipt_bytes.push(b'\n');
    write_new(&args.out, &receipt_bytes)?;
    Ok((status.into(), receipt_seal))
}

fn main() -> Result<(), Box<dyn Error>> {
    let args = parse_args()?;
    let out = args.out.clone();
    let (status, seal) = run(args)?;
    println!(
        "status={status} receipt={} seal_sha256={seal}",
        out.display()
    );
    if status != PASS {
        return Err("native Rust PLE formula control exceeded its predeclared bound".into());
    }
    Ok(())
}

//! CPU-only direct low-rank matvec parity for HGRAVS01 activation-weighted SVD.
//!
//! Proves `y = L @ (R @ x)` from the physical payload matches a one-shot dense
//! `L @ R` product used only as a parity oracle. Dense reconstruction is never
//! a production token path.

use hawking_core::model::qwen_complete_binary::{
    decode_hgravs01_dense_f32_for_parity, hgravs01_matvec_f64, parse_hgravs01_header,
};
use serde_json::{json, Value};
use sha2::{Digest, Sha256};
use std::env;
use std::fs;
use std::path::PathBuf;
use std::process;

const RESULT_SCHEMA: &str = "hawking.ascension.qwen30_hgravs01_packed_matvec_parity_result.v1";
const RESULT_STATUS: &str =
    "EARNED_HGRAVS01_CPU_NATIVE_LOW_RANK_MATVEC_PARITY_NOT_RUNTIME_OR_CAPABILITY_QUALIFIED";

struct Arguments {
    payload: PathBuf,
    payload_sha256: String,
}

fn usage() -> &'static str {
    "usage: ascension_qwen30_hgravs01_packed_matvec_parity \
        --payload ABSOLUTE_PATH --payload-sha256 SHA256"
}

fn required(value: Option<String>, flag: &str) -> Result<String, String> {
    value.ok_or_else(|| format!("missing {flag}; {}", usage()))
}

fn required_path(value: Option<PathBuf>, flag: &str) -> Result<PathBuf, String> {
    let path = value.ok_or_else(|| format!("missing {flag}; {}", usage()))?;
    if !path.is_absolute() {
        return Err(format!("{flag} must be an absolute path"));
    }
    Ok(path)
}

fn parse_arguments() -> Result<Arguments, String> {
    let mut payload = None;
    let mut payload_sha256 = None;
    let mut args = env::args().skip(1);
    while let Some(flag) = args.next() {
        let value = args
            .next()
            .ok_or_else(|| format!("missing value for {flag:?}; {}", usage()))?;
        match flag.as_str() {
            "--payload" => {
                if payload.replace(PathBuf::from(value)).is_some() {
                    return Err(format!("{flag} was supplied more than once; {}", usage()));
                }
            }
            "--payload-sha256" => {
                if payload_sha256.replace(value).is_some() {
                    return Err(format!("{flag} was supplied more than once; {}", usage()));
                }
            }
            _ => return Err(format!("unsupported option {flag:?}; {}", usage())),
        }
    }
    Ok(Arguments {
        payload: required_path(payload, "--payload")?,
        payload_sha256: required(payload_sha256, "--payload-sha256")?,
    })
}

fn fail(detail: impl AsRef<str>) -> ! {
    eprintln!(
        "HGRAVS01 packed matvec parity refused: {}",
        detail.as_ref()
    );
    process::exit(2);
}

fn sha256_hex(bytes: &[u8]) -> String {
    format!("{:x}", Sha256::digest(bytes))
}

fn main() {
    let arguments = parse_arguments().unwrap_or_else(|error| fail(error));
    let payload = fs::read(&arguments.payload).unwrap_or_else(|error| {
        fail(format!(
            "cannot read {}: {error}",
            arguments.payload.display()
        ))
    });
    let observed = sha256_hex(&payload);
    if observed != arguments.payload_sha256 {
        fail(format!(
            "payload sha256 mismatch: observed={observed} expected={}",
            arguments.payload_sha256
        ));
    }
    let header = parse_hgravs01_header(&payload).unwrap_or_else(|error| fail(error.to_string()));
    if header.shape.len() != 2 {
        fail("payload is not a rank-2 matrix");
    }
    let cols = header.matrix_shape[1];
    let input: Vec<f64> = (0..cols)
        .map(|index| {
            let phase = (index % 17) as f64;
            (phase * 0.07 - 0.5).sin()
        })
        .collect();
    let (_, native) =
        hgravs01_matvec_f64(&payload, &input).unwrap_or_else(|error| fail(error.to_string()));
    let (_, dense) = decode_hgravs01_dense_f32_for_parity(&payload)
        .unwrap_or_else(|error| fail(error.to_string()));
    let rows = header.matrix_shape[0];
    let mut oracle = Vec::with_capacity(rows);
    let mut max_abs_error = 0.0f64;
    for row in 0..rows {
        let mut sum = 0.0f64;
        for col in 0..cols {
            sum += f64::from(dense[row * cols + col]) * input[col];
        }
        max_abs_error = max_abs_error.max((sum - native[row]).abs());
        oracle.push(sum);
    }
    const TOLERANCE: f64 = 1e-4;
    if max_abs_error > TOLERANCE {
        fail(format!(
            "native low-rank matvec diverged from dense parity oracle: max_abs_error={max_abs_error} tolerance={TOLERANCE}"
        ));
    }
    let result = json!({
        "schema": RESULT_SCHEMA,
        "status": RESULT_STATUS,
        "payload_path": arguments.payload,
        "payload_sha256": observed,
        "header": {
            "shape": header.shape,
            "matrix_shape": header.matrix_shape,
            "rank": header.rank,
            "factor_bits": header.factor_bits,
            "factor_group_size": header.factor_group_size,
            "activation_capture_sha256": header.activation_capture_sha256,
        },
        "parity": {
            "rows": rows,
            "cols": cols,
            "max_abs_error": max_abs_error,
            "tolerance_max_abs": TOLERANCE,
            "native_execution": "two_stage_low_rank_matvec_L_R_x",
            "dense_reconstruction_role": "cpu_parity_oracle_only_not_token_path",
            "passed": true,
        },
        "claim_boundary": {
            "cpu_format_oracle_only": true,
            "not_device_runtime_generation_or_coherence": true,
        },
    });
    println!("{}", serde_json::to_string_pretty(&result).unwrap_or_else(|_| Value::Null.to_string()));
}

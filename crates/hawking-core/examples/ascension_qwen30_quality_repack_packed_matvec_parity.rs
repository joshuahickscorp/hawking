//! CPU-only direct-packed matvec contract for the isolated Qwen30 HQ30GR2
//! gate/up candidate.
//!
//! The candidate operator executes the embedded HQ30G1B1 packed base and then
//! accumulates only its sorted sparse FP16 corrections.  This is deliberately
//! a narrow adapter probe: it does not materialize a full matrix, open Metal,
//! load a Qwen layer, or create a runtime/server/tournament surface.

use hawking_core::model::qwen_complete_binary::{
    complete_binary_matvec_f64, parse_qwen30_quality_residual_header,
    qwen30_quality_residual_entries, qwen30_quality_residual_matvec_f64,
};
use serde_json::{json, Value};
use sha2::{Digest, Sha256};
use std::env;
use std::fs;
use std::path::{Path, PathBuf};
use std::process;

const RESULT_SCHEMA: &str =
    "hawking.ascension.qwen30_quality_repack_packed_matvec_parity_result.v1";
const RESULT_STATUS: &str =
    "EARNED_HQ30GR2_CPU_DIRECT_PACKED_MATVEC_PARITY_NOT_RUNTIME_OR_CAPABILITY_QUALIFIED";
const CONTROL_COUNT: usize = 8;

struct Arguments {
    candidate_gate: PathBuf,
    candidate_gate_sha256: String,
    control_gate: PathBuf,
    control_gate_sha256: String,
    candidate_up: PathBuf,
    candidate_up_sha256: String,
    control_up: PathBuf,
    control_up_sha256: String,
}

struct PairProbe {
    evidence: Value,
    candidate_outputs: Vec<f64>,
    control_outputs: Vec<f64>,
}

fn usage() -> &'static str {
    "usage: ascension_qwen30_quality_repack_packed_matvec_parity \\
        --candidate-gate ABSOLUTE_PATH --candidate-gate-sha256 SHA256 \\
        --control-gate ABSOLUTE_PATH --control-gate-sha256 SHA256 \\
        --candidate-up ABSOLUTE_PATH --candidate-up-sha256 SHA256 \\
        --control-up ABSOLUTE_PATH --control-up-sha256 SHA256"
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
    let mut candidate_gate = None;
    let mut candidate_gate_sha256 = None;
    let mut control_gate = None;
    let mut control_gate_sha256 = None;
    let mut candidate_up = None;
    let mut candidate_up_sha256 = None;
    let mut control_up = None;
    let mut control_up_sha256 = None;
    let mut args = env::args().skip(1);
    while let Some(flag) = args.next() {
        let value = args
            .next()
            .ok_or_else(|| format!("missing value for {flag:?}; {}", usage()))?;
        macro_rules! once_path {
            ($slot:ident) => {
                if $slot.replace(PathBuf::from(value)).is_some() {
                    return Err(format!("{flag} was supplied more than once; {}", usage()));
                }
            };
        }
        macro_rules! once_string {
            ($slot:ident) => {
                if $slot.replace(value).is_some() {
                    return Err(format!("{flag} was supplied more than once; {}", usage()));
                }
            };
        }
        match flag.as_str() {
            "--candidate-gate" => once_path!(candidate_gate),
            "--candidate-gate-sha256" => once_string!(candidate_gate_sha256),
            "--control-gate" => once_path!(control_gate),
            "--control-gate-sha256" => once_string!(control_gate_sha256),
            "--candidate-up" => once_path!(candidate_up),
            "--candidate-up-sha256" => once_string!(candidate_up_sha256),
            "--control-up" => once_path!(control_up),
            "--control-up-sha256" => once_string!(control_up_sha256),
            _ => return Err(format!("unsupported option {flag:?}; {}", usage())),
        }
    }
    Ok(Arguments {
        candidate_gate: required_path(candidate_gate, "--candidate-gate")?,
        candidate_gate_sha256: required(candidate_gate_sha256, "--candidate-gate-sha256")?,
        control_gate: required_path(control_gate, "--control-gate")?,
        control_gate_sha256: required(control_gate_sha256, "--control-gate-sha256")?,
        candidate_up: required_path(candidate_up, "--candidate-up")?,
        candidate_up_sha256: required(candidate_up_sha256, "--candidate-up-sha256")?,
        control_up: required_path(control_up, "--control-up")?,
        control_up_sha256: required(control_up_sha256, "--control-up-sha256")?,
    })
}

fn fail(detail: impl AsRef<str>) -> ! {
    eprintln!(
        "quality-repack packed matvec parity refused: {}",
        detail.as_ref()
    );
    process::exit(2);
}

fn is_sha256(value: &str) -> bool {
    value.len() == 64
        && value
            .bytes()
            .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte))
}

fn sha256_hex(bytes: &[u8]) -> String {
    let mut digest = Sha256::new();
    digest.update(bytes);
    format!("{:x}", digest.finalize())
}

fn hash_f64(values: &[f64]) -> String {
    let mut digest = Sha256::new();
    for value in values {
        digest.update(value.to_le_bytes());
    }
    format!("{:x}", digest.finalize())
}

fn canonical_regular(path: &Path, label: &str) -> Result<PathBuf, String> {
    let metadata = fs::symlink_metadata(path)
        .map_err(|error| format!("cannot stat {label} {}: {error}", path.display()))?;
    if metadata.file_type().is_symlink() || !metadata.is_file() {
        return Err(format!("{label} must be a non-symlink regular file"));
    }
    fs::canonicalize(path).map_err(|error| format!("cannot canonicalize {label}: {error}"))
}

fn read_pinned(
    path: &Path,
    expected_sha256: &str,
    label: &str,
) -> Result<(PathBuf, Vec<u8>), String> {
    if !is_sha256(expected_sha256) {
        return Err(format!("{label} expected SHA-256 must be lowercase hex"));
    }
    let canonical = canonical_regular(path, label)?;
    let payload = fs::read(&canonical).map_err(|error| format!("cannot read {label}: {error}"))?;
    let observed = sha256_hex(&payload);
    if observed != expected_sha256 {
        return Err(format!(
            "{label} SHA-256 differs: observed={observed} expected={expected_sha256}"
        ));
    }
    Ok((canonical, payload))
}

fn deterministic_input(control: usize, column: usize) -> f64 {
    let mut state = (control as u64 + 1)
        .wrapping_mul(0x9e37_79b9_7f4a_7c15)
        .wrapping_add((column as u64 + 1).wrapping_mul(0xbf58_476d_1ce4_e5b9));
    state ^= state >> 30;
    state = state.wrapping_mul(0xbf58_476d_1ce4_e5b9);
    state ^= state >> 27;
    state = state.wrapping_mul(0x94d0_49bb_1331_11eb);
    state ^= state >> 31;
    ((state % 65_521) as i64 - 32_760) as f64 / 8_192.0
}

fn max_abs(left: &[f64], right: &[f64]) -> Result<f64, String> {
    if left.len() != right.len() {
        return Err("packed matvec vectors have incompatible length".into());
    }
    Ok(left
        .iter()
        .zip(right)
        .map(|(a, b)| (a - b).abs())
        .fold(0.0, f64::max))
}

fn silu(value: f64) -> f64 {
    let bounded = value.clamp(-60.0, 60.0);
    bounded / (1.0 + (-bounded).exp())
}

fn probe_pair(
    label: &str,
    candidate_path: &Path,
    candidate_sha256: &str,
    control_path: &Path,
    control_sha256: &str,
) -> Result<PairProbe, String> {
    let (candidate_path, candidate_payload) = read_pinned(
        candidate_path,
        candidate_sha256,
        &format!("candidate {label}"),
    )?;
    let (control_path, control_payload) = read_pinned(
        control_path,
        control_sha256,
        &format!("admitted control {label}"),
    )?;
    let (header, entries) = qwen30_quality_residual_entries(&candidate_payload)
        .map_err(|error| format!("candidate {label} HQ30GR2 parse failed: {error}"))?;
    let control_header =
        hawking_core::model::qwen_complete_binary::parse_complete_binary_header(&control_payload)
            .map_err(|error| format!("admitted control {label} HQ30G1B1 parse failed: {error}"))?;
    if header.shape != control_header.shape || header.shape.len() != 2 {
        return Err(format!(
            "candidate {label} HQ30GR2 geometry differs from admitted rank-2 control"
        ));
    }
    let rows = header.shape[0];
    let columns = header.shape[1];
    let base_end = header
        .base_offset
        .checked_add(header.base_payload_bytes)
        .ok_or_else(|| format!("candidate {label} embedded base end overflows"))?;
    if candidate_payload[header.base_offset..base_end] != control_payload {
        return Err(format!(
            "candidate {label} embedded direct base differs from admitted control bytes"
        ));
    }
    let sentinel_input = vec![0.0; columns];
    if complete_binary_matvec_f64(&candidate_payload, &sentinel_input).is_ok() {
        return Err(format!(
            "candidate {label} was accepted by the direct packed control operator"
        ));
    }
    if qwen30_quality_residual_matvec_f64(&control_payload, &sentinel_input).is_ok() {
        return Err(format!(
            "admitted control {label} was accepted by the HQ30GR2 operator"
        ));
    }
    let mut candidate_outputs = Vec::with_capacity(CONTROL_COUNT * rows);
    let mut control_outputs = Vec::with_capacity(CONTROL_COUNT * rows);
    let mut residual_outputs = Vec::with_capacity(CONTROL_COUNT * rows);
    let mut input_bytes = Vec::with_capacity(CONTROL_COUNT * columns * std::mem::size_of::<f64>());
    let mut worst_error = 0.0f64;
    for control in 0..CONTROL_COUNT {
        let input = (0..columns)
            .map(|column| deterministic_input(control, column))
            .collect::<Vec<_>>();
        for value in &input {
            input_bytes.extend_from_slice(&value.to_le_bytes());
        }
        let (_, candidate) = qwen30_quality_residual_matvec_f64(&candidate_payload, &input)
            .map_err(|error| format!("candidate {label} HQ30GR2 packed matvec failed: {error}"))?;
        let (_, control_output) =
            complete_binary_matvec_f64(&control_payload, &input).map_err(|error| {
                format!("admitted control {label} direct packed matvec failed: {error}")
            })?;
        let mut residual = vec![0.0f64; rows];
        for (flat_index, correction) in &entries {
            residual[flat_index / columns] += f64::from(*correction) * input[flat_index % columns];
        }
        let expected = control_output
            .iter()
            .zip(&residual)
            .map(|(base, correction)| base + correction)
            .collect::<Vec<_>>();
        let error = max_abs(&candidate, &expected)?;
        if !error.is_finite() || error > 1e-12 {
            return Err(format!(
                "candidate {label} direct packed base-plus-residual parity differs by {error:e}"
            ));
        }
        worst_error = worst_error.max(error);
        candidate_outputs.extend(candidate);
        control_outputs.extend(control_output);
        residual_outputs.extend(residual);
    }
    let parsed = parse_qwen30_quality_residual_header(&candidate_payload)
        .map_err(|error| format!("candidate {label} HQ30GR2 header reparse failed: {error}"))?;
    Ok(PairProbe {
        evidence: json!({
            "organ": label,
            "candidate_payload": {"path": candidate_path, "sha256": candidate_sha256, "bytes": candidate_payload.len()},
            "admitted_scalar_control_payload": {"path": control_path, "sha256": control_sha256, "bytes": control_payload.len()},
            "shape": header.shape,
            "hq30gr2": {
                "magic": "HQ30GR2\\u0000",
                "base_magic": "HQ30G1B1",
                "base_payload_bytes": parsed.base_payload_bytes,
                "residual_count": parsed.residual_count,
                "indices_sha256": parsed.indices_sha256,
                "values_sha256": parsed.values_sha256,
            },
            "embedded_base_exactly_matches_admitted_control": true,
            "exact_format_refusal": {
                "direct_packed_control_operator_refuses_hq30gr2": true,
                "hq30gr2_packed_operator_refuses_direct_control": true,
            },
            "direct_packed_matvec": {
                "no_dense_weight_materialization": true,
                "deterministic_input_count": CONTROL_COUNT,
                "input_sha256_f64le": sha256_hex(&input_bytes),
                "candidate_output_sha256_f64le": hash_f64(&candidate_outputs),
                "admitted_control_output_sha256_f64le": hash_f64(&control_outputs),
                "sparse_residual_output_sha256_f64le": hash_f64(&residual_outputs),
                "max_abs_candidate_minus_control_minus_sparse_residual": worst_error,
            },
        }),
        candidate_outputs,
        control_outputs,
    })
}

fn main() {
    let arguments = parse_arguments().unwrap_or_else(|error| fail(error));
    let gate = probe_pair(
        "model.layers.0.mlp.experts.0.gate_proj.weight",
        &arguments.candidate_gate,
        &arguments.candidate_gate_sha256,
        &arguments.control_gate,
        &arguments.control_gate_sha256,
    )
    .unwrap_or_else(|error| fail(error));
    let up = probe_pair(
        "model.layers.0.mlp.experts.0.up_proj.weight",
        &arguments.candidate_up,
        &arguments.candidate_up_sha256,
        &arguments.control_up,
        &arguments.control_up_sha256,
    )
    .unwrap_or_else(|error| fail(error));
    if gate.candidate_outputs.len() != up.candidate_outputs.len()
        || gate.control_outputs.len() != up.control_outputs.len()
    {
        fail("gate/up packed matvec output geometry differs");
    }
    let candidate_swiglu = gate
        .candidate_outputs
        .iter()
        .zip(&up.candidate_outputs)
        .map(|(gate, up)| silu(*gate) * up)
        .collect::<Vec<_>>();
    let control_swiglu = gate
        .control_outputs
        .iter()
        .zip(&up.control_outputs)
        .map(|(gate, up)| silu(*gate) * up)
        .collect::<Vec<_>>();
    let delta = candidate_swiglu
        .iter()
        .zip(&control_swiglu)
        .map(|(candidate, control)| candidate - control)
        .collect::<Vec<_>>();
    let max_abs_delta = delta.iter().map(|value| value.abs()).fold(0.0, f64::max);
    if !max_abs_delta.is_finite() {
        fail("gate/up packed matvec SwiGLU comparison is non-finite");
    }
    let result = json!({
        "schema": RESULT_SCHEMA,
        "status": RESULT_STATUS,
        "mode": "cpu_only_direct_packed_base_plus_sparse_residual_matvec_v1",
        "pairs": [gate.evidence, up.evidence],
        "gate_up_swiglu": {
            "candidate_output_sha256_f64le": hash_f64(&candidate_swiglu),
            "admitted_control_output_sha256_f64le": hash_f64(&control_swiglu),
            "candidate_minus_control_sha256_f64le": hash_f64(&delta),
            "max_abs_candidate_minus_control": max_abs_delta,
            "finite": true,
        },
        "claim_boundary": {
            "cpu_only": true,
            "metal_not_opened": true,
            "direct_packed_matvec_operator_only": true,
            "not_a_full_qwen_layer_decoder_generation_hcli_or_tps_result": true,
            "not_a_capability_tg_agent_os_or_tournament_qualification": true,
            "later_candidate_full_model_integration_requires_fresh_layer_model_and_runtime_gates": true,
        },
    });
    println!(
        "{}",
        serde_json::to_string(&result).expect("result must serialize")
    );
}

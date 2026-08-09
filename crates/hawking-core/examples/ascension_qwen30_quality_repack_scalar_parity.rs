//! CPU-only scalar compatibility/parity probe for the isolated Qwen30
//! HQ30GR2 gate/up candidate.
//!
//! This executable has no model loader, Metal device, server, HCLI, benchmark,
//! or tournament authority.  Its narrow job is to prove that the two selected
//! HQ30GR2 tensors decode as their exact admitted HQ30G1B1 control bodies plus
//! their sealed sparse FP16 corrections, and that neither direction permits a
//! silent format fallback.

use hawking_core::model::qwen_complete_binary::{
    decode_complete_binary_f32, decode_qwen30_quality_residual_f32,
    qwen30_quality_residual_entries, Qwen30QualityResidualHeader,
};
use serde_json::{json, Value};
use sha2::{Digest, Sha256};
use std::collections::BTreeMap;
use std::env;
use std::fs;
use std::path::{Path, PathBuf};
use std::process;

const RESULT_SCHEMA: &str = "hawking.ascension.qwen30_quality_repack_scalar_parity_result.v1";
const RESULT_STATUS: &str =
    "EARNED_HQ30GR2_CPU_SCALAR_COMPATIBILITY_PARITY_NOT_RUNTIME_OR_CAPABILITY_QUALIFIED";
const CONTROL_COUNT: usize = 4;

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
    candidate_projection: Vec<f64>,
    control_projection: Vec<f64>,
}

fn usage() -> &'static str {
    "usage: ascension_qwen30_quality_repack_scalar_parity \\
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
    eprintln!("quality-repack scalar parity refused: {}", detail.as_ref());
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

fn hash_f64(values: &[f64]) -> String {
    let mut digest = Sha256::new();
    for value in values {
        digest.update(value.to_le_bytes());
    }
    format!("{:x}", digest.finalize())
}

fn deterministic_activation(control: usize, column: usize) -> f64 {
    // A compact fully specified CPU-only activation basis.  It is intentionally
    // unrelated to a model prompt or source teacher; it only tests the scalar
    // adapter identity W_candidate = W_control + residual.
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

fn projection(values: &[f32], shape: &[usize]) -> Result<Vec<f64>, String> {
    if shape.len() != 2
        || shape[0] == 0
        || shape[1] == 0
        || shape[0].checked_mul(shape[1]) != Some(values.len())
    {
        return Err("scalar parity requires a non-empty rank-2 gate/up tensor".into());
    }
    let rows = shape[0];
    let cols = shape[1];
    let mut output = Vec::with_capacity(CONTROL_COUNT * rows);
    for control in 0..CONTROL_COUNT {
        for row in 0..rows {
            let mut sum = 0.0f64;
            let offset = row * cols;
            for column in 0..cols {
                sum +=
                    deterministic_activation(control, column) * f64::from(values[offset + column]);
            }
            output.push(sum);
        }
    }
    Ok(output)
}

fn max_abs_difference(left: &[f64], right: &[f64]) -> Result<f64, String> {
    if left.len() != right.len() {
        return Err("parity vectors have different lengths".into());
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
    let (_, candidate_values) = decode_qwen30_quality_residual_f32(&candidate_payload)
        .map_err(|error| format!("candidate {label} HQ30GR2 scalar decode failed: {error}"))?;
    let (control_header, control_values) =
        decode_complete_binary_f32(&control_payload).map_err(|error| {
            format!("admitted control {label} HQ30G1B1 scalar decode failed: {error}")
        })?;
    if header.shape != control_header.shape || header.base.elements != control_header.elements {
        return Err(format!(
            "candidate {label} HQ30GR2 geometry differs from admitted control"
        ));
    }
    let base_end = header
        .base_offset
        .checked_add(header.base_payload_bytes)
        .ok_or_else(|| format!("candidate {label} embedded base end overflows"))?;
    if candidate_payload[header.base_offset..base_end] != control_payload {
        return Err(format!(
            "candidate {label} embedded direct base is not the admitted control payload"
        ));
    }
    if decode_complete_binary_f32(&candidate_payload).is_ok() {
        return Err(format!(
            "candidate {label} incorrectly accepted a direct-layout fallback"
        ));
    }
    if decode_qwen30_quality_residual_f32(&control_payload).is_ok() {
        return Err(format!(
            "admitted control {label} incorrectly accepted a residual-layout fallback"
        ));
    }
    if candidate_values.len() != control_values.len() {
        return Err(format!(
            "candidate {label} decoded value count differs from control"
        ));
    }
    let corrections = entries.iter().copied().collect::<BTreeMap<_, _>>();
    if corrections.len() != entries.len() || corrections.is_empty() {
        return Err(format!(
            "candidate {label} residual entries are not non-empty unique corrections"
        ));
    }
    let mut nonzero_delta_count = 0usize;
    for (index, (candidate, control)) in candidate_values.iter().zip(&control_values).enumerate() {
        let expected = control + corrections.get(&index).copied().unwrap_or(0.0);
        if candidate.to_bits() != expected.to_bits() {
            return Err(format!("candidate {label} scalar value {index} does not equal control plus exact FP16 correction"));
        }
        if candidate.to_bits() != control.to_bits() {
            nonzero_delta_count += 1;
        }
    }
    if nonzero_delta_count != corrections.len() {
        return Err(format!(
            "candidate {label} residual changed {nonzero_delta_count} values, expected {}",
            corrections.len()
        ));
    }
    let candidate_projection = projection(&candidate_values, &header.shape)?;
    let control_projection = projection(&control_values, &header.shape)?;
    let residual_values = candidate_values
        .iter()
        .zip(&control_values)
        .map(|(candidate, control)| candidate - control)
        .collect::<Vec<_>>();
    let residual_projection = projection(&residual_values, &header.shape)?;
    let observed_projection_delta = candidate_projection
        .iter()
        .zip(&control_projection)
        .map(|(candidate, control)| candidate - control)
        .collect::<Vec<_>>();
    let max_abs_projection_parity_error =
        max_abs_difference(&observed_projection_delta, &residual_projection)?;
    if !max_abs_projection_parity_error.is_finite() || max_abs_projection_parity_error > 1e-10 {
        return Err(format!("candidate {label} scalar projection parity differs by {max_abs_projection_parity_error:e}"));
    }
    let header_evidence = residual_header_evidence(&header);
    Ok(PairProbe {
        evidence: json!({
            "organ": label,
            "candidate_payload": {"path": candidate_path, "sha256": candidate_sha256, "bytes": candidate_payload.len()},
            "admitted_scalar_control_payload": {"path": control_path, "sha256": control_sha256, "bytes": control_payload.len()},
            "shape": header.shape,
            "hq30gr2": header_evidence,
            "embedded_base_exactly_matches_admitted_control": true,
            "exact_fallback_refusal": {
                "direct_decoder_refuses_hq30gr2": true,
                "hq30gr2_decoder_refuses_direct_control": true,
            },
            "scalar_identity": {
                "candidate_equals_admitted_control_plus_exact_sparse_fp16_residual": true,
                "changed_scalar_count": nonzero_delta_count,
                "residual_entry_count": corrections.len(),
            },
            "projection_parity": {
                "deterministic_control_count": CONTROL_COUNT,
                "activation_family": "splitmix64_column_basis_v1",
                "candidate_output_sha256_f64le": hash_f64(&candidate_projection),
                "control_output_sha256_f64le": hash_f64(&control_projection),
                "residual_output_sha256_f64le": hash_f64(&residual_projection),
                "max_abs_candidate_minus_control_minus_residual": max_abs_projection_parity_error,
            },
        }),
        candidate_projection,
        control_projection,
    })
}

fn residual_header_evidence(header: &Qwen30QualityResidualHeader) -> Value {
    json!({
        "magic": "HQ30GR2\\u0000",
        "version": 1,
        "base_magic": "HQ30G1B1",
        "base_payload_bytes": header.base_payload_bytes,
        "residual_count": header.residual_count,
        "indices_sha256": header.indices_sha256,
        "values_sha256": header.values_sha256,
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
    if gate.candidate_projection.len() != up.candidate_projection.len()
        || gate.control_projection.len() != up.control_projection.len()
    {
        fail("gate/up scalar projection geometry differs");
    }
    let candidate_swiglu = gate
        .candidate_projection
        .iter()
        .zip(&up.candidate_projection)
        .map(|(gate, up)| silu(*gate) * up)
        .collect::<Vec<_>>();
    let control_swiglu = gate
        .control_projection
        .iter()
        .zip(&up.control_projection)
        .map(|(gate, up)| silu(*gate) * up)
        .collect::<Vec<_>>();
    let swiglu_delta = candidate_swiglu
        .iter()
        .zip(&control_swiglu)
        .map(|(candidate, control)| candidate - control)
        .collect::<Vec<_>>();
    let max_abs_swiglu_delta = swiglu_delta
        .iter()
        .map(|value| value.abs())
        .fold(0.0, f64::max);
    if !max_abs_swiglu_delta.is_finite() {
        fail("candidate/control SwiGLU scalar comparison is non-finite");
    }
    let result = json!({
        "schema": RESULT_SCHEMA,
        "status": RESULT_STATUS,
        "mode": "cpu_only_scalar_adapter_compatibility_parity_v1",
        "pairs": [gate.evidence, up.evidence],
        "gate_up_swiglu": {
            "deterministic_control_count": CONTROL_COUNT,
            "candidate_output_sha256_f64le": hash_f64(&candidate_swiglu),
            "admitted_control_output_sha256_f64le": hash_f64(&control_swiglu),
            "candidate_minus_control_sha256_f64le": hash_f64(&swiglu_delta),
            "max_abs_candidate_minus_control": max_abs_swiglu_delta,
            "finite": true,
        },
        "claim_boundary": {
            "cpu_only": true,
            "metal_not_opened": true,
            "not_a_full_qwen_layer_decoder_generation_hcli_or_tps_result": true,
            "not_a_capability_tg_agent_os_or_tournament_qualification": true,
            "candidate_only_scalar_adapter_contract": true,
        },
    });
    println!(
        "{}",
        serde_json::to_string(&result).expect("result must serialize")
    );
}

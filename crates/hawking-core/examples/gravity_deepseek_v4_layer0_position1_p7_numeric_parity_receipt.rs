//! Seal a conservative, source-bound receipt for a fresh P7 v3 diagnostic.
//!
//! This program deliberately cannot turn the bounded graph into exact-storage,
//! full-layer, runtime, generation, HCLI, or TPS evidence.  It accepts only a
//! fresh v3 create-new diagnostic and independently revalidates its artifact,
//! component-receipt, physical-trace, accounting, and scoped Numeric Parity
//! V2.1 boundaries before atomically publishing a new sealed receipt.
//!
//! ```sh
//! cargo run --release -p hawking-core --example gravity_deepseek_v4_layer0_position1_p7_numeric_parity_receipt -- \
//!   --artifact /absolute/path/to/full-43-layer-stream.gravity \
//!   --diagnostic /absolute/path/to/DSV4F_LAYER0_POSITION1_P7_DEVICE_UNSEALED-v3.json \
//!   --p4b-receipt /absolute/path/to/DSV4F_P4B_POSITION1_COMPLETE_ATTENTION_METAL-v1.json \
//!   --p1-ffn-receipt /absolute/path/to/DSV4F_LAYER0_POSITION1_FULL_FFN_CPU_ORACLE-v1.json \
//!   --p6a-receipt /absolute/path/to/DSV4F_LAYER0_MOE_METAL_P6A-v1.json \
//!   --out /absolute/path/to/DSV4F_P7_LAYER0_POSITION1_NUMERIC_PARITY_V2_1-v1.json
//! ```

use hawking_core::gravity_deepseek_v4::DeepSeekV4FullStreamReader;
use serde_json::{json, Value};
use sha2::{Digest, Sha256};
use std::error::Error;
use std::fs::{self, File, OpenOptions};
use std::io::Write;
use std::path::{Path, PathBuf};

const RECEIPT_SCHEMA: &str =
    "hawking.gravity.deepseek_v4.p7_layer0_position1_device_numeric_parity_v2_1.v1";
const RECEIPT_STATUS: &str =
    "PASS_REAL_METAL_P7_LAYER0_POSITION1_NUMERIC_PARITY_V2_1_ONLY_NOT_EXACT_STORAGE_NOT_RUNTIME";
const DIAGNOSTIC_SCHEMA: &str =
    "hawking.gravity.deepseek_v4.p7_layer0_position1_device_diagnostic.v3";
const DIAGNOSTIC_STATUS: &str = "P7_LAYER0_POSITION1_REAL_METAL_DEVICE_GRAPH_UNSEALED_DIAGNOSTIC_V3_NUMERIC_PARITY_V2_1_ONLY_NOT_EXACT_STORAGE_NOT_RUNTIME";
const P4B_RECEIPT_SCHEMA: &str =
    "hawking.gravity.deepseek_v4.p4b_position1_complete_attention_metal.v1";
const P4B_RECEIPT_STATUS: &str =
    "PASS_REAL_METAL_P4B_POSITION1_COMPLETE_ATTENTION_PARITY_NOT_RUNTIME";
const P1_FFN_RECEIPT_SCHEMA: &str =
    "hawking.gravity.deepseek_v4.layer0_position1_full_ffn_cpu_oracle.v1";
const P1_FFN_RECEIPT_STATUS: &str =
    "PASS_SOURCE_DERIVED_CPU_LAYER0_POSITION1_FULL_FFN_NOT_INDEPENDENT_UPSTREAM_RUNTIME_PARITY";
const P6A_RECEIPT_SCHEMA: &str =
    "hawking.gravity.deepseek_v4.layer0_moe_metal_p6a_full_route_wave.v1";
const P6A_RECEIPT_STATUS: &str =
    "PASS_REAL_METAL_DEVICE_GATE_ROUTE_FULL_SIX_EXPERT_WAVE_NOT_FULL_RUNTIME";
const EXPECTED_ROUTE_IDS_BY_TOP_SLOT: [u64; 6] = [72, 168, 184, 142, 174, 177];
const EXPECTED_NUMERIC_COMBINE_ORDER: [(u64, u64); 6] =
    [(0, 72), (3, 142), (1, 168), (4, 174), (5, 177), (2, 184)];

type ReceiptResult<T> = Result<T, Box<dyn Error>>;

struct Args {
    artifact: PathBuf,
    diagnostic: PathBuf,
    p4b_receipt: PathBuf,
    p1_ffn_receipt: PathBuf,
    p6a_receipt: PathBuf,
    out: Option<PathBuf>,
    validate_only: bool,
}

#[derive(Clone)]
struct ReceiptBinding {
    path: PathBuf,
    file_sha256: String,
    seal_sha256: String,
    schema: String,
    status: String,
    transitive_p1_attention_seal_sha256: Option<String>,
}

fn main() -> ReceiptResult<()> {
    let args = parse_args()?;
    let reader = DeepSeekV4FullStreamReader::admit(&args.artifact)?;
    let p4b_binding = bind_sealed_component(
        &reader,
        &args.p4b_receipt,
        "DSV4F_P4B_POSITION1_COMPLETE_ATTENTION_METAL-v1.json",
        P4B_RECEIPT_SCHEMA,
        P4B_RECEIPT_STATUS,
        true,
    )?;
    let p1_ffn_binding = bind_sealed_component(
        &reader,
        &args.p1_ffn_receipt,
        "DSV4F_LAYER0_POSITION1_FULL_FFN_CPU_ORACLE-v1.json",
        P1_FFN_RECEIPT_SCHEMA,
        P1_FFN_RECEIPT_STATUS,
        false,
    )?;
    let p6a_binding = bind_sealed_component(
        &reader,
        &args.p6a_receipt,
        "DSV4F_LAYER0_MOE_METAL_P6A-v1.json",
        P6A_RECEIPT_SCHEMA,
        P6A_RECEIPT_STATUS,
        false,
    )?;
    let (diagnostic_path, diagnostic_raw, diagnostic) = read_diagnostic(&args.diagnostic)?;
    validate_diagnostic(
        &reader,
        &diagnostic,
        &p4b_binding,
        &p1_ffn_binding,
        &p6a_binding,
    )?;

    if args.validate_only {
        println!(
            "P7 v3 diagnostic validation passed; no receipt written: {}",
            diagnostic_path.display()
        );
        return Ok(());
    }

    let out = args
        .out
        .as_ref()
        .ok_or("--out is required unless --validate-only")?;
    let actual_sidecars = value_at(&diagnostic, &["post_completion_device_diagnostics"])?;
    let actual_topology = value_at(&diagnostic, &["actual_graph_topology"])?;
    let run_provenance = value_at(&diagnostic, &["run_provenance"])?;
    let run_accounting = value_at(&diagnostic, &["run_accounting"])?;
    let source_controls = value_at(&diagnostic, &["source_controls"])?;
    let p6_residency = value_at(&diagnostic, &["p6_residency"])?;
    let source_code_provenance = value_at(&diagnostic, &["source_code_provenance"])?;
    let scope = value_at(&diagnostic, &["scope"])?;
    let diagnostic_canonical_unsigned_sha256 =
        text_at(&diagnostic, &["canonical_unsigned_sha256"])?;

    let unsigned = json!({
        "schema": RECEIPT_SCHEMA,
        "status": RECEIPT_STATUS,
        "receipt_classification": {
            "numeric_parity": "NUMERIC_PARITY_V2_1_ONLY_NOT_RUNTIME",
            "exact_storage": false,
            "full_decoder_layer_parity": false,
            "full_model_or_causal_runtime": false,
            "generated_token": false,
            "hcli_endpoint": false,
            "base_true_tps": false,
            "component_receipts_are_not_direct_numeric_ancestry": true,
            "statement": "This receipt seals only a bounded real-Metal layer-0/token-19923/position-1 same-actual-input Numeric Parity V2.1 diagnostic. P4B remains Numeric Parity V2.1-only; no exact-storage upgrade is admitted.",
        },
        "artifact": artifact_binding_json(&reader),
        "source": {
            "repository": reader.source_identity().repository,
            "revision": reader.source_identity().revision,
            "source_parent_retained": false,
            "runner_build_and_source_provenance_observed_at_capture": source_code_provenance,
            "source_controls_reverified_from_admitted_stream_before_sealing": true,
            "all_listed_control_reads_touched_chunk_sha256_verified_before_use": true,
        },
        "predecessors": {
            "p4b_bounded_attention_component": {
                "receipt": receipt_binding_json(&p4b_binding),
                "classification_required_from_fresh_device_output": "P4B_NUMERIC_PARITY_V2_1_ONLY",
                "exact_storage": false,
                "relation": "component/input-state provenance only; not direct exact-storage or full-stage numerical ancestry",
                "transitive_position1_complete_attention_cpu_oracle_seal_sha256": p4b_binding.transitive_p1_attention_seal_sha256.clone(),
            },
            "p1_full_ffn_cpu_oracle": {
                "receipt": receipt_binding_json(&p1_ffn_binding),
                "relation": "source-semantic component anchor only; not the direct numerical parent of the captured actual-input P7 graph",
            },
            "p6a_full_route_wave": {
                "receipt": receipt_binding_json(&p6a_binding),
                "relation": "device component/topology reference only; its deterministic fixture is not the captured graph's actual FFn-norm input and is not direct numerical ancestry",
            },
        },
        "unsealed_input": {
            "path": diagnostic_path.display().to_string(),
            "file_sha256": sha256(&diagnostic_raw),
            "schema": DIAGNOSTIC_SCHEMA,
            "status": DIAGNOSTIC_STATUS,
            "unsealed": true,
            "canonical_unsigned_sha256": diagnostic_canonical_unsigned_sha256,
            "validated_before_receipt_publication": true,
        },
        "run_provenance": run_provenance,
        "run_accounting": run_accounting,
        "source_controls": source_controls,
        "p6_residency": p6_residency,
        "actual_graph_topology": actual_topology,
        "actual_input_sidecars": actual_sidecars,
        "numeric_parity_v2_1": {
            "schema": "hawking.numeric_parity.v2_1",
            "conditional_predecessor_classification": "P4B_NUMERIC_PARITY_V2_1_ONLY",
            "mhc_pre_controls_pass": true,
            "p6_route_controls_pass": true,
            "p6_moe_body_store_pass": true,
            "mhc_post_child_store_pass": true,
            "source_f32_device_storage_bit_identity_recorded_separately_from_fp64_numeric_authority": true,
            "fp64_projected_storage_bit_identity_is_diagnostic_only_and_not_an_exact-storage_gate": true,
            "detailed_same_actual_input_scores_and_storage_observations": "retained verbatim in actual_input_sidecars",
        },
        "scope": scope,
        "claim_boundary": "This sealed receipt proves a single fresh real-Metal, physically attributed bounded P4B->P7->P6->P7 graph at layer 0, token ID 19923, position 1, with 5 command buffers, 46 encoders, 96 dispatches, zero recorded in-graph host handoffs, zero recorded fallback paths, and the retained same-actual-input Numeric Parity V2.1 sidecars. It does not prove exact storage parity, a complete decoder layer, a 43-layer causal runtime, first token, continuation, HCLI, or BASE_TRUE_TPS.",
    });
    let (receipt, seal_sha256) = seal(decimal_strings(unsigned))?;
    write_new_receipt(out, &receipt)?;
    println!(
        "sealed P7 Numeric Parity V2.1-only receipt {}\nseal_sha256={seal_sha256}",
        out.display()
    );
    Ok(())
}

fn parse_args() -> ReceiptResult<Args> {
    let mut artifact = None;
    let mut diagnostic = None;
    let mut p4b_receipt = None;
    let mut p1_ffn_receipt = None;
    let mut p6a_receipt = None;
    let mut out = None;
    let mut validate_only = false;
    let mut args = std::env::args_os().skip(1);
    while let Some(flag) = args.next() {
        match flag.to_string_lossy().as_ref() {
            "--artifact" => artifact = args.next().map(PathBuf::from),
            "--diagnostic" => diagnostic = args.next().map(PathBuf::from),
            "--p4b-receipt" => p4b_receipt = args.next().map(PathBuf::from),
            "--p1-ffn-receipt" => p1_ffn_receipt = args.next().map(PathBuf::from),
            "--p6a-receipt" => p6a_receipt = args.next().map(PathBuf::from),
            "--out" => out = args.next().map(PathBuf::from),
            "--validate-only" => validate_only = true,
            "--help" | "-h" => return Err(usage().into()),
            other => return Err(format!("unknown argument {other}").into()),
        }
    }
    let artifact = artifact.ok_or("--artifact is required")?;
    let diagnostic = diagnostic.ok_or("--diagnostic is required")?;
    let p4b_receipt = p4b_receipt.ok_or("--p4b-receipt is required")?;
    let p1_ffn_receipt = p1_ffn_receipt.ok_or("--p1-ffn-receipt is required")?;
    let p6a_receipt = p6a_receipt.ok_or("--p6a-receipt is required")?;
    if !artifact.is_absolute()
        || !diagnostic.is_absolute()
        || !p4b_receipt.is_absolute()
        || !p1_ffn_receipt.is_absolute()
        || !p6a_receipt.is_absolute()
        || out.as_ref().is_some_and(|path| !path.is_absolute())
    {
        return Err("all paths must be absolute".into());
    }
    if !validate_only && out.is_none() {
        return Err("--out is required unless --validate-only is set".into());
    }
    if validate_only && out.is_some() {
        return Err("--validate-only cannot be combined with --out".into());
    }
    Ok(Args {
        artifact,
        diagnostic,
        p4b_receipt,
        p1_ffn_receipt,
        p6a_receipt,
        out,
        validate_only,
    })
}

fn usage() -> &'static str {
    "usage: gravity_deepseek_v4_layer0_position1_p7_numeric_parity_receipt \\\n+  --artifact <absolute full Gravity dir> \\\n+  --diagnostic <absolute P7 v3 unsealed diagnostic.json> \\\n+  --p4b-receipt <absolute P4B receipt.json> \\\n+  --p1-ffn-receipt <absolute P1 FFn CPU receipt.json> \\\n+  --p6a-receipt <absolute P6A receipt.json> \\\n+  [--validate-only | --out <absolute new sealed receipt.json>]"
}

fn read_diagnostic(input: &Path) -> ReceiptResult<(PathBuf, Vec<u8>, Value)> {
    if input.file_name().and_then(|name| name.to_str())
        != Some("DSV4F_LAYER0_POSITION1_P7_DEVICE_UNSEALED-v3.json")
    {
        return Err(
            "receipt producer accepts only the fixed fresh P7 v3 diagnostic basename".into(),
        );
    }
    if fs::symlink_metadata(input)?.file_type().is_symlink() {
        return Err("diagnostic symlinks are not admitted".into());
    }
    let path = fs::canonicalize(input)?;
    if !fs::metadata(&path)?.is_file() {
        return Err("diagnostic is not a regular file".into());
    }
    let raw = fs::read(&path)?;
    let value: Value = serde_json::from_slice(&raw)?;
    Ok((path, raw, value))
}

fn bind_sealed_component(
    reader: &DeepSeekV4FullStreamReader,
    input: &Path,
    expected_basename: &str,
    expected_schema: &str,
    expected_status: &str,
    require_transitive_p1_attention: bool,
) -> ReceiptResult<ReceiptBinding> {
    if input.file_name().and_then(|name| name.to_str()) != Some(expected_basename) {
        return Err(
            format!("wrong component receipt basename; expected {expected_basename}").into(),
        );
    }
    if fs::symlink_metadata(input)?.file_type().is_symlink() {
        return Err("component receipt symlinks are not admitted".into());
    }
    let path = fs::canonicalize(input)?;
    if !fs::metadata(&path)?.is_file() {
        return Err("component receipt is not a regular file".into());
    }
    let raw = fs::read(&path)?;
    let value: Value = serde_json::from_slice(&raw)?;
    verify_sealed_json(&value, expected_basename)?;
    let schema = text_at(&value, &["schema"])?;
    let status = text_at(&value, &["status"])?;
    if schema != expected_schema || status != expected_status {
        return Err(format!(
            "component receipt schema/status mismatch: observed={schema}/{status}"
        )
        .into());
    }
    if text_at(&value, &["artifact", "manifest_seal_sha256"])? != reader.manifest_seal_sha256()
        || text_at(&value, &["artifact", "manifest_file_sha256"])? != reader.manifest_file_sha256()
    {
        return Err("component receipt artifact binding differs from the admitted stream".into());
    }
    let transitive_p1_attention_seal_sha256 = if require_transitive_p1_attention {
        let predecessor = value_at(&value, &["predecessors", "position1_complete_cpu_oracle"])?;
        if Path::new(text_at(predecessor, &["path"])?)
            .file_name()
            .and_then(|name| name.to_str())
            != Some("DSV4F_LAYER0_POSITION1_COMPLETE_ATTENTION_CPU_ORACLE-v1.json")
        {
            return Err("P4B binding omits the expected transitive P1 attention receipt".into());
        }
        let file_sha256 = text_at(predecessor, &["file_sha256"])?;
        let seal_sha256 = text_at(predecessor, &["seal_sha256"])?;
        if !is_sha256(file_sha256) || !is_sha256(seal_sha256) {
            return Err("P4B transitive P1 attention hashes are malformed".into());
        }
        Some(seal_sha256.to_owned())
    } else {
        None
    };
    Ok(ReceiptBinding {
        path,
        file_sha256: sha256(&raw),
        seal_sha256: text_at(&value, &["seal_sha256"])?.to_owned(),
        schema: schema.to_owned(),
        status: status.to_owned(),
        transitive_p1_attention_seal_sha256,
    })
}

fn validate_diagnostic(
    reader: &DeepSeekV4FullStreamReader,
    diagnostic: &Value,
    p4b_binding: &ReceiptBinding,
    p1_ffn_binding: &ReceiptBinding,
    p6a_binding: &ReceiptBinding,
) -> ReceiptResult<()> {
    if diagnostic.get("seal_sha256").is_some() {
        return Err("P7 v3 diagnostic must remain unsealed input, not a nested receipt".into());
    }
    expect_text(diagnostic, &["schema"], DIAGNOSTIC_SCHEMA)?;
    expect_text(diagnostic, &["status"], DIAGNOSTIC_STATUS)?;
    expect_bool(diagnostic, &["unsealed"], true)?;
    verify_diagnostic_canonical_hash(diagnostic)?;
    validate_artifact_binding(reader, diagnostic)?;
    validate_component_bindings(diagnostic, p4b_binding, p1_ffn_binding, p6a_binding)?;
    validate_run_provenance(diagnostic)?;
    validate_run_accounting(diagnostic)?;
    validate_topology(diagnostic)?;
    validate_p4b_boundary(diagnostic)?;
    validate_source_controls(reader, diagnostic)?;
    validate_p6_residency(diagnostic)?;
    validate_actual_input_sidecars(diagnostic)?;
    Ok(())
}

fn verify_diagnostic_canonical_hash(diagnostic: &Value) -> ReceiptResult<()> {
    let recorded = text_at(diagnostic, &["canonical_unsigned_sha256"])?;
    if !is_sha256(recorded) {
        return Err("diagnostic canonical_unsigned_sha256 is malformed".into());
    }
    let mut unsigned = diagnostic.clone();
    unsigned
        .as_object_mut()
        .ok_or("diagnostic root is not an object")?
        .remove("canonical_unsigned_sha256")
        .ok_or("diagnostic lacks canonical_unsigned_sha256")?;
    let observed = sha256(&canonical_json(&unsigned));
    if observed != recorded {
        return Err(format!(
            "diagnostic canonical unsigned hash mismatch: recorded={recorded} observed={observed}"
        )
        .into());
    }
    Ok(())
}

fn validate_artifact_binding(
    reader: &DeepSeekV4FullStreamReader,
    diagnostic: &Value,
) -> ReceiptResult<()> {
    let artifact = value_at(diagnostic, &["artifact"])?;
    if text_at(artifact, &["path"])? != reader.artifact_root().display().to_string()
        || text_at(artifact, &["manifest_seal_sha256"])? != reader.manifest_seal_sha256()
        || text_at(artifact, &["manifest_file_sha256"])? != reader.manifest_file_sha256()
        || text_at(artifact, &["restart_receipt_seal_sha256"])? != reader.restart_seal_sha256()
        || text_at(artifact, &["source_repository"])? != reader.source_identity().repository
        || text_at(artifact, &["source_revision"])? != reader.source_identity().revision
    {
        return Err(
            "diagnostic artifact/source identity does not match current admitted stream".into(),
        );
    }
    expect_bool(artifact, &["source_parent_retained"], false)
}

fn validate_component_bindings(
    diagnostic: &Value,
    p4b_binding: &ReceiptBinding,
    p1_ffn_binding: &ReceiptBinding,
    p6a_binding: &ReceiptBinding,
) -> ReceiptResult<()> {
    let bindings = value_at(diagnostic, &["component_receipt_bindings"])?;
    expect_bool(
        bindings,
        &["all_are_component_or_topology_bindings_not_direct_numeric_ancestry"],
        true,
    )?;
    compare_binding(
        value_at(bindings, &["p4b_bounded_attention_predecessor", "receipt"])?,
        p4b_binding,
        "diagnostic P4B binding",
    )?;
    compare_binding(
        value_at(bindings, &["p1_full_ffn_cpu_oracle", "receipt"])?,
        p1_ffn_binding,
        "diagnostic P1 FFn binding",
    )?;
    compare_binding(
        value_at(bindings, &["p6a_full_route_wave", "receipt"])?,
        p6a_binding,
        "diagnostic P6A binding",
    )?;
    let transitive = text_at(
        bindings,
        &[
            "p4b_bounded_attention_predecessor",
            "transitive_position1_complete_attention_cpu_oracle_seal_sha256",
        ],
    )?;
    if p4b_binding.transitive_p1_attention_seal_sha256.as_deref() != Some(transitive) {
        return Err("diagnostic P4B transitive P1 attention seal differs from independently validated P4B receipt".into());
    }
    for path in [
        &["p4b_bounded_attention_predecessor", "relation"][..],
        &["p1_full_ffn_cpu_oracle", "relation"][..],
        &["p6a_full_route_wave", "relation"][..],
    ] {
        if text_at(bindings, path)?.is_empty() {
            return Err("diagnostic component relation is empty".into());
        }
    }
    Ok(())
}

fn validate_run_provenance(diagnostic: &Value) -> ReceiptResult<()> {
    let run = value_at(diagnostic, &["run_provenance"])?;
    let nonce = text_at(run, &["run_nonce_sha256"])?;
    if !is_sha256(nonce) {
        return Err("diagnostic run nonce is not SHA-256".into());
    }
    let started = decimal_u128(text_at(run, &["run_started_unix_ns_text"])?)?;
    let finished = decimal_u128(text_at(run, &["run_finished_unix_ns_text"])?)?;
    if finished < started {
        return Err("diagnostic run finished before it started".into());
    }
    if u64_at(run, &["process_id"])? == 0 {
        return Err("diagnostic process ID is zero".into());
    }
    let executable = value_at(run, &["executable"])?;
    let executable_path = PathBuf::from(text_at(executable, &["path"])?);
    if !executable_path.is_absolute()
        || fs::symlink_metadata(&executable_path)?
            .file_type()
            .is_symlink()
    {
        return Err("captured executable path is not an admitted non-symlink absolute path".into());
    }
    let canonical_executable = fs::canonicalize(&executable_path)?;
    let metadata = fs::metadata(&canonical_executable)?;
    if !metadata.is_file() || metadata.len() == 0 {
        return Err("captured executable is no longer a nonempty regular file".into());
    }
    if sha256(&fs::read(&canonical_executable)?) != text_at(executable, &["sha256"])?
        || metadata.len() != u64_at(executable, &["bytes"])?
    {
        return Err("captured executable hash/size no longer matches fresh diagnostic".into());
    }
    if !matches!(
        text_at(executable, &["build_profile"])?,
        "release" | "debug"
    ) || text_at(executable, &["cargo_package"])?.is_empty()
        || text_at(executable, &["cargo_package_version"])?.is_empty()
    {
        return Err("captured executable provenance is incomplete".into());
    }
    let platform = value_at(run, &["host_platform"])?;
    for key in [
        "operating_system",
        "architecture",
        "kernel_release",
        "macos_product_version",
        "macos_build_version",
    ] {
        if text_at(platform, &[key])?.is_empty() {
            return Err("captured host platform field is empty".into());
        }
    }
    let trace = value_at(run, &["physical_trace"])?;
    if !is_sha256(text_at(trace, &["interval_id"])?)
        || text_at(trace, &["phase"])? != "dsv4f_p7_layer0_position1"
        || text_at(trace, &["role"])? != "p4b_p7_p6_p7_bounded_graph"
        || u64_at(trace, &["batch"])? != 1
        || u64_at(trace, &["iteration"])? != 1
        || u64_at(trace, &["command_buffers"])? != 5
        || u64_at(trace, &["compute_encoders"])? != 46
    {
        return Err("physical trace provenance is incomplete or mismatched".into());
    }
    let source = value_at(diagnostic, &["source_code_provenance"])?;
    if !is_git_revision(text_at(source, &["checkout_revision"])?)
        || !is_sha256(text_at(source, &["worktree_porcelain_sha256"])?)
        || !is_sha256(text_at(source, &["p7_shader_embedded_sha256"])?)
    {
        return Err("captured source-code provenance has malformed hashes".into());
    }
    expect_bool(
        source,
        &["p7_shader_embedded_matches_current_source_file"],
        true,
    )?;
    let source_files = value_at(source, &["source_files_sha256"])?
        .as_object()
        .ok_or("captured source file hashes are not an object")?;
    for name in [
        "examples/gravity_deepseek_v4_layer0_position1_p7_device.rs",
        "src/gravity_deepseek_v4_p7_device.rs",
        "src/gravity_deepseek_v4_p7_composition.rs",
        "src/gravity_deepseek_v4_p6_device.rs",
        "src/gravity_deepseek_v4_p4b_device.rs",
        "src/metal/mod.rs",
        "shaders/deepseek_v4_p7.metal",
    ] {
        let hash = source_files
            .get(name)
            .and_then(Value::as_str)
            .ok_or("captured source file hash missing")?;
        if !is_sha256(hash) {
            return Err("captured source file hash malformed".into());
        }
    }
    Ok(())
}

fn validate_run_accounting(diagnostic: &Value) -> ReceiptResult<()> {
    let accounting = value_at(diagnostic, &["run_accounting"])?;
    for key in [
        "in_graph_host_handoffs",
        "activation_host_handoffs",
        "route_host_handoffs",
        "kv_state_host_handoffs",
        "fallback_paths",
    ] {
        if u64_at(accounting, &[key])? != 0 {
            return Err(format!("diagnostic {key} must be zero").into());
        }
    }
    if u64_at(accounting, &["source_control_staging_reads_before_graph"])? != 4
        || u64_at(accounting, &["source_control_staging_bytes_before_graph"])? == 0
        || u64_at(accounting, &["post_completion_diagnostic_readbacks"])? != 15
        || u64_at(accounting, &["post_completion_diagnostic_readback_bytes"])? != 92_408
    {
        return Err(
            "diagnostic source-staging or post-completion readback accounting changed".into(),
        );
    }
    expect_bool(
        accounting,
        &["post_completion_readbacks_are_not_graph_handoffs"],
        true,
    )?;
    if text_at(accounting, &["fallback_policy"])?.is_empty() {
        return Err("diagnostic fallback policy is empty".into());
    }
    Ok(())
}

fn validate_topology(diagnostic: &Value) -> ReceiptResult<()> {
    let topology = value_at(diagnostic, &["actual_graph_topology"])?;
    for (key, expected) in [
        ("command_buffers", 5),
        ("cpu_visible_completion_waits", 5),
        ("gpu_dispatches", 96),
        ("compute_encoders", 46),
        ("physical_trace_command_buffers", 5),
        ("physical_trace_compute_encoders", 46),
        ("trace_samples", 5),
    ] {
        if u64_at(topology, &[key])? != expected {
            return Err(format!("diagnostic topology {key} differs from bounded graph").into());
        }
    }
    if u64_at(topology, &["buffers_created_during_graph"])? != 4
        || u64_at(topology, &["bytes_allocated_during_graph"])? != 41_008
    {
        return Err("diagnostic graph allocation accounting changed".into());
    }
    for (stage, command_buffers, waits, dispatches, encoders) in [
        ("p4b", 1, 1, 33, 33),
        ("p6", 2, 2, 60, 10),
        ("p7_owned", 2, 2, 3, 3),
    ] {
        let value = value_at(topology, &[stage])?;
        if u64_at(value, &["command_buffers"])? != command_buffers
            || u64_at(value, &["cpu_visible_completion_waits"])? != waits
            || u64_at(value, &["gpu_dispatches"])? != dispatches
            || u64_at(value, &["compute_encoders"])? != encoders
        {
            return Err(format!("diagnostic {stage} topology changed").into());
        }
    }
    let batches = value_at(topology, &["ordered_command_batches"])?
        .as_array()
        .ok_or("ordered command batches are not an array")?;
    let expected_stages = [
        "P4B position-1 complete attention",
        "P7 mHC-FFN pre plus FFn RMSNorm",
        "P6 Gate/route/W1-W3/cast/SwiGLU",
        "P6 down-QAT/W2/cast/source-order combine",
        "P7 mHC-FFN post",
    ];
    if batches.len() != expected_stages.len() {
        return Err("diagnostic has the wrong command-batch count".into());
    }
    for (index, (batch, expected_stage)) in batches.iter().zip(expected_stages).enumerate() {
        if u64_at(batch, &["ordinal"])? != index as u64
            || text_at(batch, &["stage"])? != expected_stage
            || text_at(batch, &["kernel_name"])? != "dispatch_batch"
            || u64_at(batch, &["host_wall_us"])? == 0
        {
            return Err("diagnostic command-batch identity changed".into());
        }
        validate_optional_gpu_timing(batch)?;
    }
    Ok(())
}

fn validate_optional_gpu_timing(batch: &Value) -> ReceiptResult<()> {
    let duration = value_at(batch, &["gpu_duration_us"])?;
    let start = value_at(batch, &["gpu_start_ns"])?;
    let end = value_at(batch, &["gpu_end_ns"])?;
    match (duration.as_u64(), start.as_u64(), end.as_u64()) {
        (Some(duration), Some(start), Some(end)) if duration > 0 && end > start => Ok(()),
        (None, None, None) if duration.is_null() && start.is_null() && end.is_null() => Ok(()),
        _ => Err("GPU timing must be a complete valid triplet or explicit nulls".into()),
    }
}

fn validate_p4b_boundary(diagnostic: &Value) -> ReceiptResult<()> {
    let p4b = value_at(diagnostic, &["p4b_predecessor"])?;
    expect_text(p4b, &["classification"], "P4B_NUMERIC_PARITY_V2_1_ONLY")?;
    expect_bool(p4b, &["exact_storage"], false)?;
    if text_at(p4b, &["policy"])?.is_empty() {
        return Err("P4B predecessor policy is empty".into());
    }
    Ok(())
}

fn validate_source_controls(
    reader: &DeepSeekV4FullStreamReader,
    diagnostic: &Value,
) -> ReceiptResult<()> {
    let controls = value_at(diagnostic, &["source_controls"])?;
    let expected = [
        ("ffn_norm", "layers.0.ffn_norm.weight", "BF16", vec![4096]),
        ("hc_ffn_fn", "layers.0.hc_ffn_fn", "F32", vec![24, 16384]),
        ("hc_ffn_base", "layers.0.hc_ffn_base", "F32", vec![24]),
        ("hc_ffn_scale", "layers.0.hc_ffn_scale", "F32", vec![3]),
    ];
    for (key, name, dtype, shape) in expected {
        let binding = value_at(controls, &[key])?;
        let metadata = reader.tensor_metadata(name)?;
        if text_at(binding, &["name"])? != name
            || text_at(binding, &["dtype"])? != dtype
            || metadata.dtype != dtype
            || metadata.shape != shape
            || value_at(binding, &["shape"])? != &serde_json::to_value(&shape)?
            || u64_at(binding, &["bytes"])? != metadata.bytes
        {
            return Err(
                format!("diagnostic source control {key} geometry differs from stream").into(),
            );
        }
        let bytes = usize::try_from(metadata.bytes)
            .map_err(|_| "control tensor byte count exceeds usize")?;
        let verified = reader.read_verified_full(name, bytes)?;
        if sha256(&verified) != text_at(binding, &["sha256"])? {
            return Err(format!("diagnostic source control {key} hash differs from stream").into());
        }
    }
    if text_at(controls, &["staging"])?
        != "direct verified reader controls; no fabricated PreparedDecodeInput position"
    {
        return Err("diagnostic source-control staging boundary changed".into());
    }
    Ok(())
}

fn validate_p6_residency(diagnostic: &Value) -> ReceiptResult<()> {
    let residency = value_at(diagnostic, &["p6_residency"])?;
    if u64_at(residency, &["hot_capacity_bytes"])? != 80_216_064
        || u64_at(residency, &["hot_resident_bytes"])? != 80_216_064
        || u64_at(residency, &["cold_resident_bytes"])? != 0
        || u64_at(residency, &["source_bundle_loads"])? != 6
        || u64_at(residency, &["source_payload_bytes_returned"])? != 80_216_064
    {
        return Err("diagnostic P6 residency accounting changed".into());
    }
    if u64_array_at(residency, &["top_slot_route_ids"])? != EXPECTED_ROUTE_IDS_BY_TOP_SLOT {
        return Err("diagnostic P6 top-slot route IDs changed".into());
    }
    let order = value_at(residency, &["numeric_source_combine_order"])?
        .as_array()
        .ok_or("diagnostic P6 combine order is not an array")?;
    if order.len() != EXPECTED_NUMERIC_COMBINE_ORDER.len() {
        return Err("diagnostic P6 combine order length changed".into());
    }
    for (value, (slot, expert)) in order.iter().zip(EXPECTED_NUMERIC_COMBINE_ORDER) {
        let pair = value
            .as_array()
            .ok_or("diagnostic P6 combine entry is not an array")?;
        if pair.len() != 2 || pair[0].as_u64() != Some(slot) || pair[1].as_u64() != Some(expert) {
            return Err("diagnostic P6 numeric source combine order changed".into());
        }
    }
    let hot_keys = value_at(residency, &["hot_expert_keys"])?
        .as_array()
        .ok_or("diagnostic P6 hot keys are not an array")?;
    if hot_keys.len() != 6 {
        return Err("diagnostic P6 hot key count changed".into());
    }
    for (value, expert) in hot_keys.iter().zip([72, 142, 168, 174, 177, 184]) {
        if u64_at(value, &["layer"])? != 0 || u64_at(value, &["expert"])? != expert {
            return Err("diagnostic P6 hot key order changed".into());
        }
    }
    Ok(())
}

fn validate_actual_input_sidecars(diagnostic: &Value) -> ReceiptResult<()> {
    let sidecars = value_at(diagnostic, &["post_completion_device_diagnostics"])?;
    validate_observed_bf16(
        sidecars,
        "actual_p4b_attention_hc_post_bf16",
        16_384,
        "87042c82bacedc0f4981b5194d846400a388da684779cd7bc0a87683666d1dc5",
    )?;
    validate_observed_bf16(
        sidecars,
        "actual_ffn_norm_bf16",
        4096,
        "ac7840658678e2c39532c8c1c4453daea6430b99d5a43359bd4b510350539b1d",
    )?;
    validate_observed_bf16(
        sidecars,
        "moe_output_bf16",
        4096,
        "21f10dad7f5642af790e16ec31173538248c7e8dbbfd4b057e3ba9086bee87d7",
    )?;
    validate_observed_bf16(
        sidecars,
        "child_hc_state_bf16",
        16_384,
        "4380b270cf13583e5ab2f60e99a132b824f7276bb5aef9cdd56fc219d6ed5723",
    )?;
    let mhc = value_at(sidecars, &["same_actual_input_mhc_controls"])?;
    expect_bool(
        mhc,
        &["numeric_parity_v2_1", "all_scored_mhc_controls_pass"],
        true,
    )?;
    expect_text(
        mhc,
        &["numeric_parity_v2_1", "schema"],
        "hawking.numeric_parity.v2_1",
    )?;
    validate_storage_delta(
        mhc,
        &["source_f32_vs_device_storage", "ffn_reduced_bf16"],
        true,
    )?;
    validate_storage_delta(mhc, &["fp64_projected_storage", "ffn_reduced_bf16"], false)?;
    let route = value_at(sidecars, &["same_actual_input_numeric_parity_v2_1"])?;
    expect_bool(route, &["continuous_and_discrete_controls_pass"], true)?;
    expect_bool(route, &["exact_gate_logit_bits"], true)?;
    expect_bool(route, &["exact_tid2eid_ids"], true)?;
    expect_text(
        route,
        &["gate_logits", "schema"],
        "hawking.numeric_parity.v2_1",
    )?;
    expect_text(
        route,
        &["original_scores", "schema"],
        "hawking.numeric_parity.v2_1",
    )?;
    expect_text(
        route,
        &["selected_weights", "schema"],
        "hawking.numeric_parity.v2_1",
    )?;
    if u64_array_at(route, &["route_valid_word"])? != [1] {
        return Err("same-actual-input P6 route-valid word changed".into());
    }
    let cpu_route = value_at(sidecars, &["same_actual_input_cpu_route"])?;
    expect_bool(cpu_route, &["route_ids", "bit_exact"], true)?;
    if u64_array_at(cpu_route, &["route_ids", "observed"])? != EXPECTED_ROUTE_IDS_BY_TOP_SLOT {
        return Err("same-actual-input CPU route IDs changed".into());
    }
    let moe = value_at(sidecars, &["same_actual_input_moe_body"])?;
    expect_bool(
        moe,
        &["numeric_parity_v2_1", "source_f32_and_device_score_pass"],
        true,
    )?;
    expect_text(
        moe,
        &["numeric_parity_v2_1", "schema"],
        "hawking.numeric_parity.v2_1",
    )?;
    expect_bool(moe, &["source_numeric_combine_order", "all_match"], true)?;
    validate_combine_order(moe, &["source_numeric_combine_order", "device_p6"])?;
    validate_combine_order(moe, &["source_numeric_combine_order", "source_f32"])?;
    validate_combine_order(moe, &["source_numeric_combine_order", "fp64_authority"])?;
    validate_storage_delta(
        moe,
        &["source_f32_vs_device_storage", "moe_output_bf16"],
        true,
    )?;
    validate_fp64_storage_delta(moe, &["fp64_projected_storage", "moe_output_bf16"])?;
    let child = value_at(sidecars, &["same_actual_input_mhc_post_child"])?;
    expect_bool(
        child,
        &["numeric_parity_v2_1", "source_f32_and_device_score_pass"],
        true,
    )?;
    expect_text(
        child,
        &["numeric_parity_v2_1", "schema"],
        "hawking.numeric_parity.v2_1",
    )?;
    validate_storage_delta(
        child,
        &["source_f32_vs_device_storage", "child_hc_state_bf16"],
        true,
    )?;
    validate_storage_delta(
        child,
        &["fp64_projected_storage", "child_hc_state_bf16"],
        false,
    )?;
    Ok(())
}

fn validate_observed_bf16(
    sidecars: &Value,
    key: &str,
    expected_count: u64,
    expected_sha256: &str,
) -> ReceiptResult<()> {
    let observed = value_at(sidecars, &[key])?;
    if u64_at(observed, &["element_count"])? != expected_count
        || text_at(observed, &["observed_sha256"])? != expected_sha256
    {
        return Err(format!("observed device BF16 hash changed for {key}").into());
    }
    Ok(())
}

fn validate_storage_delta(value: &Value, path: &[&str], expected_exact: bool) -> ReceiptResult<()> {
    let delta = value_at(value, path)?;
    expect_bool(delta, &["bit_exact"], expected_exact)?;
    if u64_at(delta, &["element_count"])? == 0
        || !is_sha256(text_at(delta, &["expected_sha256"])?)
        || !is_sha256(text_at(delta, &["observed_sha256"])?)
    {
        return Err("storage delta lacks valid hash/count evidence".into());
    }
    Ok(())
}

fn validate_fp64_storage_delta(value: &Value, path: &[&str]) -> ReceiptResult<()> {
    let delta = value_at(value, path)?;
    if !value_at(delta, &["bit_exact"])?.is_boolean()
        || u64_at(delta, &["element_count"])? == 0
        || !is_sha256(text_at(delta, &["expected_sha256"])?)
        || !is_sha256(text_at(delta, &["observed_sha256"])?)
    {
        return Err("FP64 projected storage delta lacks valid diagnostic evidence".into());
    }
    Ok(())
}

fn validate_combine_order(value: &Value, path: &[&str]) -> ReceiptResult<()> {
    let order = value_at(value, path)?
        .as_array()
        .ok_or("source combine order is not an array")?;
    if order.len() != EXPECTED_NUMERIC_COMBINE_ORDER.len() {
        return Err("source combine order length changed".into());
    }
    for (entry, (slot, expert)) in order.iter().zip(EXPECTED_NUMERIC_COMBINE_ORDER) {
        let pair = entry
            .as_array()
            .ok_or("source combine entry is not an array")?;
        if pair.len() != 2 || pair[0].as_u64() != Some(slot) || pair[1].as_u64() != Some(expert) {
            return Err("source combine order changed".into());
        }
    }
    Ok(())
}

fn artifact_binding_json(reader: &DeepSeekV4FullStreamReader) -> Value {
    json!({
        "path": reader.artifact_root().display().to_string(),
        "manifest_seal_sha256": reader.manifest_seal_sha256(),
        "manifest_file_sha256": reader.manifest_file_sha256(),
        "restart_receipt_seal_sha256": reader.restart_seal_sha256(),
        "source_repository": reader.source_identity().repository,
        "source_revision": reader.source_identity().revision,
        "source_parent_retained": false,
    })
}

fn receipt_binding_json(binding: &ReceiptBinding) -> Value {
    json!({
        "path": binding.path.display().to_string(),
        "file_sha256": binding.file_sha256,
        "seal_sha256": binding.seal_sha256,
        "schema": binding.schema,
        "status": binding.status,
    })
}

fn compare_binding(value: &Value, expected: &ReceiptBinding, label: &str) -> ReceiptResult<()> {
    if text_at(value, &["path"])? != expected.path.display().to_string()
        || text_at(value, &["file_sha256"])? != expected.file_sha256
        || text_at(value, &["seal_sha256"])? != expected.seal_sha256
        || text_at(value, &["schema"])? != expected.schema
        || text_at(value, &["status"])? != expected.status
    {
        return Err(
            format!("{label} differs from independently verified component receipt").into(),
        );
    }
    Ok(())
}

fn verify_sealed_json(value: &Value, label: &str) -> ReceiptResult<()> {
    let recorded = text_at(value, &["seal_sha256"])?;
    if !is_sha256(recorded) {
        return Err(format!("{label} has malformed seal_sha256").into());
    }
    let mut unsigned = value.clone();
    unsigned
        .as_object_mut()
        .ok_or_else(|| format!("{label} root is not an object"))?
        .remove("seal_sha256")
        .ok_or_else(|| format!("{label} lacks seal_sha256"))?;
    if sha256(&canonical_json(&unsigned)) != recorded {
        return Err(format!("{label} canonical seal mismatch").into());
    }
    Ok(())
}

fn value_at<'a>(value: &'a Value, path: &[&str]) -> ReceiptResult<&'a Value> {
    let mut current = value;
    for key in path {
        current = current
            .get(*key)
            .ok_or_else(|| format!("JSON field missing: {}", path.join(".")))?;
    }
    Ok(current)
}

fn text_at<'a>(value: &'a Value, path: &[&str]) -> ReceiptResult<&'a str> {
    value_at(value, path)?
        .as_str()
        .ok_or_else(|| format!("JSON field is not text: {}", path.join(".")).into())
}

fn u64_at(value: &Value, path: &[&str]) -> ReceiptResult<u64> {
    value_at(value, path)?
        .as_u64()
        .ok_or_else(|| format!("JSON field is not an unsigned integer: {}", path.join(".")).into())
}

fn u64_array_at(value: &Value, path: &[&str]) -> ReceiptResult<Vec<u64>> {
    value_at(value, path)?
        .as_array()
        .ok_or_else(|| format!("JSON field is not an array: {}", path.join(".")))?
        .iter()
        .map(|value| {
            value
                .as_u64()
                .ok_or("JSON array value is not an unsigned integer".into())
        })
        .collect()
}

fn decimal_u128(value: &str) -> ReceiptResult<u128> {
    if value.is_empty() || !value.bytes().all(|byte| byte.is_ascii_digit()) {
        return Err("timestamp text is not unsigned decimal".into());
    }
    Ok(value.parse()?)
}

fn expect_text(value: &Value, path: &[&str], expected: &str) -> ReceiptResult<()> {
    if text_at(value, path)? != expected {
        return Err(format!("JSON field differs: {}", path.join(".")).into());
    }
    Ok(())
}

fn expect_bool(value: &Value, path: &[&str], expected: bool) -> ReceiptResult<()> {
    if value_at(value, path)?.as_bool() != Some(expected) {
        return Err(format!("JSON boolean differs: {}", path.join(".")).into());
    }
    Ok(())
}

fn is_sha256(value: &str) -> bool {
    value.len() == 64
        && value
            .bytes()
            .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte))
}

fn is_git_revision(value: &str) -> bool {
    matches!(value.len(), 40 | 64)
        && value
            .bytes()
            .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte))
}

fn sha256(bytes: &[u8]) -> String {
    format!("{:x}", Sha256::digest(bytes))
}

fn seal(mut receipt: Value) -> ReceiptResult<(Value, String)> {
    if !receipt.is_object() || receipt.get("seal_sha256").is_some() {
        return Err("receipt must be an unsealed JSON object".into());
    }
    let seal_sha256 = sha256(&canonical_json(&receipt));
    receipt
        .as_object_mut()
        .expect("receipt object was checked")
        .insert("seal_sha256".to_owned(), Value::String(seal_sha256.clone()));
    Ok((receipt, seal_sha256))
}

fn write_new_receipt(path: &Path, receipt: &Value) -> ReceiptResult<()> {
    if path.exists() {
        return Err(format!("refusing to overwrite sealed receipt {}", path.display()).into());
    }
    let parent = path
        .parent()
        .filter(|parent| !parent.as_os_str().is_empty())
        .ok_or("sealed receipt output requires a parent directory")?;
    fs::create_dir_all(parent)?;
    let name = path
        .file_name()
        .and_then(|name| name.to_str())
        .ok_or("sealed receipt output filename is not UTF-8")?;
    let temporary = parent.join(format!(".{name}.{}.p7-receipt.tmp", std::process::id()));
    if temporary.exists() {
        return Err(format!(
            "sealed receipt temporary already exists {}",
            temporary.display()
        )
        .into());
    }
    let mut file = OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(&temporary)?;
    if let Err(error) = file
        .write_all(&serde_json::to_vec_pretty(receipt)?)
        .and_then(|_| file.write_all(b"\n"))
        .and_then(|_| file.sync_all())
    {
        let _ = fs::remove_file(&temporary);
        return Err(Box::new(error));
    }
    drop(file);
    if let Err(error) = fs::hard_link(&temporary, path) {
        let _ = fs::remove_file(&temporary);
        return Err(format!("atomically publish sealed receipt: {error}").into());
    }
    fs::remove_file(&temporary)?;
    File::open(parent)?.sync_all()?;
    Ok(())
}

fn canonical_json(value: &Value) -> Vec<u8> {
    let mut output = Vec::new();
    write_canonical_json(&mut output, value);
    output
}

fn write_canonical_json(output: &mut Vec<u8>, value: &Value) {
    match value {
        Value::Null => output.extend_from_slice(b"null"),
        Value::Bool(true) => output.extend_from_slice(b"true"),
        Value::Bool(false) => output.extend_from_slice(b"false"),
        Value::Number(number) => output.extend_from_slice(number.to_string().as_bytes()),
        Value::String(string) => output.extend_from_slice(
            serde_json::to_string(string)
                .expect("JSON string serialization is infallible")
                .as_bytes(),
        ),
        Value::Array(values) => {
            output.push(b'[');
            for (index, value) in values.iter().enumerate() {
                if index != 0 {
                    output.push(b',');
                }
                write_canonical_json(output, value);
            }
            output.push(b']');
        }
        Value::Object(values) => {
            let mut keys = values.keys().collect::<Vec<_>>();
            keys.sort();
            output.push(b'{');
            for (index, key) in keys.into_iter().enumerate() {
                if index != 0 {
                    output.push(b',');
                }
                output.extend_from_slice(
                    serde_json::to_string(key)
                        .expect("JSON key serialization is infallible")
                        .as_bytes(),
                );
                output.push(b':');
                write_canonical_json(output, &values[key]);
            }
            output.push(b'}');
        }
    }
}

fn decimal_strings(value: Value) -> Value {
    match value {
        Value::Number(number) if number.is_i64() || number.is_u64() => Value::Number(number),
        Value::Number(number) => Value::String(number.to_string()),
        Value::Array(values) => Value::Array(values.into_iter().map(decimal_strings).collect()),
        Value::Object(values) => Value::Object(
            values
                .into_iter()
                .map(|(key, value)| (key, decimal_strings(value)))
                .collect(),
        ),
        other => other,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn canonical_json_sorts_keys_and_decimal_strings_only_converts_floats() {
        let value = json!({"z": 1.25, "a": {"b": 2, "a": true}});
        assert_eq!(
            String::from_utf8(canonical_json(&decimal_strings(value))).unwrap(),
            "{\"a\":{\"a\":true,\"b\":2},\"z\":\"1.25\"}"
        );
    }

    #[test]
    fn optional_gpu_timing_rejects_mixed_nulls() {
        let mixed = json!({"gpu_duration_us": null, "gpu_start_ns": 1, "gpu_end_ns": 2});
        assert!(validate_optional_gpu_timing(&mixed).is_err());
        let absent = json!({"gpu_duration_us": null, "gpu_start_ns": null, "gpu_end_ns": null});
        assert!(validate_optional_gpu_timing(&absent).is_ok());
    }
}

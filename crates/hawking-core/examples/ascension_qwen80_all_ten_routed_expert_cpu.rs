//! Source-bound Qwen3-Coder-Next all-ten routed-expert CPU oracle.
//!
//! This is one generic host for the route-0..9 wave, not ten one-off expert
//! examples.  It admits the complete direct-packed artifact once, binds the
//! immutable descriptor-only all-ten plan to the sealed router capture, then
//! evaluates every selected `gate -> up -> SiLU*up -> down -> route weight`
//! body through the compact CPU reference path.
//!
//! It deliberately stops before the shared expert, routed/shared combine,
//! residual, full layer, token, decoder, generation, HCLI, or TPS.  The
//! emitted `future_device_graph` is an unexecuted capture specification only;
//! no Metal context or GPU lease is used by this binary.

use hawking_core::model::qwen80_complete_runtime::{
    Qwen80AllTenRoutedExpertPlanAuthority, Qwen80CompleteArtifactCatalog, Qwen80RouteSelection,
};
use hawking_core::model::qwen_complete_binary::{
    decode_complete_binary_f32, CompleteBinaryAdmission, QwenCompleteBinaryModel,
};
use serde_json::{json, Value};
use sha2::{Digest, Sha256};
use std::env;
use std::fs::{self, OpenOptions};
use std::io::Write;
use std::path::{Path, PathBuf};
use std::process;

const RESULT_SCHEMA: &str = "hawking.ascension.qwen80_all_ten_routed_expert_cpu_oracle.v1";
const RESULT_STATUS: &str =
    "EARNED_CURRENT_ADMITTED_QWEN80_ALL_TEN_ROUTED_EXPERT_CPU_ORACLE_READY_FOR_SEPARATE_DEVICE_LEASE";
const ROUTER_OUTER_SCHEMA: &str =
    "hawking.ascension.qwen80_direct_packed_postnorm_router_top10_outer_launcher.v1";
const ROUTER_OUTER_STATUS: &str =
    "CAPTURED_QWEN80_POSTNORM_ROUTER_TOP10_OUTER_TERMINAL_COMPONENT_ONLY";
const ROUTER_INNER_SCHEMA: &str = "hawking.ascension.qwen80_direct_packed_postnorm_router_top10.v1";
const ROUTER_INNER_STATUS: &str =
    "EARNED_CURRENT_ADMITTED_QWEN80_DIRECT_PACKED_POSTNORM_ROUTER_TOP10_STRICT_MATH_METAL_COMPONENT_NOT_COMPLETE_LAYER_OR_TOKEN";
const HIDDEN: usize = 2_048;
const TOP_K: usize = 10;
const RMS_EPSILON: f32 = 1.0e-6;

#[derive(Debug)]
struct Args {
    manifest: PathBuf,
    expected_manifest_seal_sha256: String,
    expected_source_audit_seal_sha256: String,
    expected_source_revision: String,
    route_plan: PathBuf,
    router_inner: PathBuf,
    router_outer: PathBuf,
    capture_dir: PathBuf,
}

fn usage() -> &'static str {
    "usage: ascension_qwen80_all_ten_routed_expert_cpu \\
        --manifest ABSOLUTE_PATH \\
        --expected-manifest-seal-sha256 SHA256 \\
        --expected-source-audit-seal-sha256 SHA256 \\
        --expected-source-revision REVISION \\
        --route-plan ABSOLUTE_PATH \\
        --router-inner ABSOLUTE_PATH \\
        --router-outer ABSOLUTE_PATH \\
        --capture-dir NEW_ABSOLUTE_DIRECTORY"
}

fn parse_args() -> Result<Args, String> {
    let mut manifest = None;
    let mut expected_manifest_seal_sha256 = None;
    let mut expected_source_audit_seal_sha256 = None;
    let mut expected_source_revision = None;
    let mut route_plan = None;
    let mut router_inner = None;
    let mut router_outer = None;
    let mut capture_dir = None;
    let mut args = env::args().skip(1);
    while let Some(flag) = args.next() {
        let value = args
            .next()
            .ok_or_else(|| format!("missing value for {flag:?}; {}", usage()))?;
        let destination = match flag.as_str() {
            "--manifest" => &mut manifest,
            "--expected-manifest-seal-sha256" => &mut expected_manifest_seal_sha256,
            "--expected-source-audit-seal-sha256" => &mut expected_source_audit_seal_sha256,
            "--expected-source-revision" => &mut expected_source_revision,
            "--route-plan" => &mut route_plan,
            "--router-inner" => &mut router_inner,
            "--router-outer" => &mut router_outer,
            "--capture-dir" => &mut capture_dir,
            _ => return Err(format!("unsupported argument {flag:?}; {}", usage())),
        };
        if destination.replace(value).is_some() {
            return Err(format!("argument {flag:?} was repeated; {}", usage()));
        }
    }
    let absolute = |value: Option<String>, label: &str| -> Result<PathBuf, String> {
        let path = PathBuf::from(value.ok_or_else(|| format!("missing {label}; {}", usage()))?);
        if !path.is_absolute() {
            return Err(format!("{label} must be absolute"));
        }
        Ok(path)
    };
    let capture_dir = absolute(capture_dir, "--capture-dir")?;
    if !capture_dir.parent().is_some_and(Path::is_dir) {
        return Err("--capture-dir parent must already exist".into());
    }
    Ok(Args {
        manifest: absolute(manifest, "--manifest")?,
        expected_manifest_seal_sha256: expected_manifest_seal_sha256
            .ok_or_else(|| format!("missing --expected-manifest-seal-sha256; {}", usage()))?,
        expected_source_audit_seal_sha256: expected_source_audit_seal_sha256
            .ok_or_else(|| format!("missing --expected-source-audit-seal-sha256; {}", usage()))?,
        expected_source_revision: expected_source_revision
            .ok_or_else(|| format!("missing --expected-source-revision; {}", usage()))?,
        route_plan: absolute(route_plan, "--route-plan")?,
        router_inner: absolute(router_inner, "--router-inner")?,
        router_outer: absolute(router_outer, "--router-outer")?,
        capture_dir,
    })
}

fn sha256_hex(bytes: &[u8]) -> String {
    format!("{:x}", Sha256::digest(bytes))
}

fn canonical_sha256(value: &str, label: &str) -> Result<(), String> {
    if value.len() != 64
        || !value.bytes().all(|byte| {
            byte.is_ascii_digit() || (byte.is_ascii_lowercase() && byte.is_ascii_hexdigit())
        })
    {
        return Err(format!("{label} is not a lowercase SHA-256"));
    }
    Ok(())
}

fn read_regular_json(path: &Path, label: &str) -> Result<(Vec<u8>, Value), String> {
    let metadata = fs::symlink_metadata(path)
        .map_err(|error| format!("{label} metadata failed at {}: {error}", path.display()))?;
    if !metadata.file_type().is_file() || metadata.file_type().is_symlink() {
        return Err(format!(
            "{label} is not a regular non-symlink file: {}",
            path.display()
        ));
    }
    let bytes = fs::read(path).map_err(|error| format!("{label} read failed: {error}"))?;
    let value =
        serde_json::from_slice(&bytes).map_err(|error| format!("{label} JSON invalid: {error}"))?;
    Ok((bytes, value))
}

fn object<'a>(value: &'a Value, field: &str, label: &str) -> Result<&'a Value, String> {
    value
        .get(field)
        .filter(|value| value.is_object())
        .ok_or_else(|| format!("{label} missing object {field:?}"))
}

fn string<'a>(value: &'a Value, field: &str, label: &str) -> Result<&'a str, String> {
    value
        .get(field)
        .and_then(Value::as_str)
        .filter(|value| !value.is_empty())
        .ok_or_else(|| format!("{label} missing string {field:?}"))
}

fn boolean(value: &Value, field: &str, label: &str) -> Result<bool, String> {
    value
        .get(field)
        .and_then(Value::as_bool)
        .ok_or_else(|| format!("{label} missing boolean {field:?}"))
}

fn source_route(
    value: &Value,
    ids_field: &str,
    weights_field: &str,
    label: &str,
) -> Result<Qwen80RouteSelection, String> {
    let ids = value
        .get(ids_field)
        .and_then(Value::as_array)
        .ok_or_else(|| format!("{label} missing ids {ids_field:?}"))?;
    let weights = value
        .get(weights_field)
        .and_then(Value::as_array)
        .ok_or_else(|| format!("{label} missing weights {weights_field:?}"))?;
    if ids.len() != TOP_K || weights.len() != TOP_K {
        return Err(format!("{label} does not contain exactly {TOP_K} routes"));
    }
    let mut parsed_ids = [0u16; TOP_K];
    let mut parsed_weights = [0.0f32; TOP_K];
    for index in 0..TOP_K {
        parsed_ids[index] = ids[index]
            .as_u64()
            .and_then(|value| u16::try_from(value).ok())
            .ok_or_else(|| format!("{label} route {index} has invalid expert id"))?;
        parsed_weights[index] = weights[index]
            .as_f64()
            .filter(|value| value.is_finite())
            .map(|value| value as f32)
            .filter(|value| value.is_finite())
            .ok_or_else(|| format!("{label} route {index} has invalid weight"))?;
    }
    let route = Qwen80RouteSelection {
        ids: parsed_ids,
        weights: parsed_weights,
    };
    route.validate().map_err(|error| error.to_string())?;
    Ok(route)
}

fn f32_sha256(values: &[f32]) -> Result<String, String> {
    if values.len() != HIDDEN || values.iter().any(|value| !value.is_finite()) {
        return Err(format!("expected {HIDDEN} finite f32 values for hash"));
    }
    let mut hasher = Sha256::new();
    for value in values {
        hasher.update(value.to_bits().to_le_bytes());
    }
    Ok(format!("{:x}", hasher.finalize()))
}

fn deterministic_router_fixture_residual() -> Vec<f32> {
    let mut state = 0x6e9f_5b9d_bf58_3ebd_u64;
    (0..HIDDEN)
        .map(|index| {
            state = state
                .wrapping_mul(6_364_136_223_846_793_005)
                .wrapping_add(1_442_695_040_888_963_407);
            let unit = ((state >> 40) & 0x00ff_ffff) as f32 / 16_777_215.0;
            let phase = ((index * 43 % 101) as f32 - 50.0) / 173.0;
            (unit * 2.0 - 1.0) + phase
        })
        .collect()
}

fn normalize_router_fixture(catalog: &Qwen80CompleteArtifactCatalog) -> Result<Vec<f32>, String> {
    let payload = catalog
        .verified_direct_tensor_payload("model.layers.0.post_attention_layernorm.weight")
        .map_err(|error| error.to_string())?;
    let (_header, weights) =
        decode_complete_binary_f32(&payload).map_err(|error| error.to_string())?;
    let residual = deterministic_router_fixture_residual();
    if weights.len() != HIDDEN {
        return Err("admitted post-attention RMSNorm has wrong width".into());
    }
    let mean_square = residual.iter().map(|value| value * value).sum::<f32>() / HIDDEN as f32;
    let inverse = (mean_square + RMS_EPSILON).sqrt().recip();
    let normalized = residual
        .iter()
        .zip(weights)
        .map(|(&value, weight)| value * inverse * (1.0 + weight))
        .collect::<Vec<_>>();
    if normalized.iter().any(|value| !value.is_finite()) {
        return Err("deterministic router normalized hidden is non-finite".into());
    }
    Ok(normalized)
}

fn write_capture(capture_dir: &Path, name: &str, bytes: &[u8]) -> Result<(), String> {
    let path = capture_dir.join(name);
    let temporary = capture_dir.join(format!(".{name}.{}.tmp", process::id()));
    let mut file = OpenOptions::new()
        .create_new(true)
        .write(true)
        .open(&temporary)
        .map_err(|error| {
            format!(
                "cannot create capture temporary {}: {error}",
                temporary.display()
            )
        })?;
    file.write_all(bytes)
        .and_then(|_| file.sync_all())
        .map_err(|error| {
            format!(
                "cannot write capture temporary {}: {error}",
                temporary.display()
            )
        })?;
    fs::rename(&temporary, &path)
        .map_err(|error| format!("cannot publish capture {}: {error}", path.display()))
}

fn run(args: &Args) -> Result<Value, String> {
    let (manifest_bytes, _manifest) = read_regular_json(&args.manifest, "complete manifest")?;
    let manifest_document_sha256 = sha256_hex(&manifest_bytes);
    let (plan_bytes, plan_document) = read_regular_json(&args.route_plan, "all-ten route plan")?;
    let plan_document_sha256 = sha256_hex(&plan_bytes);
    let (inner_bytes, router_inner) =
        read_regular_json(&args.router_inner, "router inner receipt")?;
    let router_receipt_sha256 = sha256_hex(&inner_bytes);
    let (outer_bytes, router_outer) =
        read_regular_json(&args.router_outer, "router outer receipt")?;
    let router_outer_receipt_sha256 = sha256_hex(&outer_bytes);

    if string(&router_outer, "schema", "router outer")? != ROUTER_OUTER_SCHEMA
        || string(&router_outer, "status", "router outer")? != ROUTER_OUTER_STATUS
    {
        return Err(
            "router outer receipt schema/status is not the sealed component terminal".into(),
        );
    }
    let outer_seal = string(&router_outer, "seal_sha256", "router outer")?;
    canonical_sha256(outer_seal, "router outer seal")?;
    let outer_inner = object(&router_outer, "inner_probe_capture", "router outer")?;
    if string(outer_inner, "path", "router outer inner binding")?
        != args
            .router_inner
            .canonicalize()
            .map_err(|error| error.to_string())?
            .to_string_lossy()
        || string(outer_inner, "sha256", "router outer inner binding")? != router_receipt_sha256
        || !boolean(outer_inner, "metal_performed", "router outer inner binding")?
    {
        return Err(
            "router outer receipt does not bind the supplied successful inner component".into(),
        );
    }
    if string(&router_inner, "schema", "router inner")? != ROUTER_INNER_SCHEMA
        || string(&router_inner, "status", "router inner")? != ROUTER_INNER_STATUS
        || string(&router_inner, "mode", "router inner")? != "metal"
        || !boolean(
            &router_inner,
            "metal_device_or_dispatch_performed",
            "router inner",
        )?
    {
        return Err("router inner receipt is not a strict-Metal component receipt".into());
    }
    let router_binding = object(&router_inner, "artifact_binding", "router inner")?;
    if string(
        router_binding,
        "manifest_document_sha256",
        "router inner binding",
    )? != manifest_document_sha256
        || string(
            router_binding,
            "manifest_seal_sha256",
            "router inner binding",
        )? != args.expected_manifest_seal_sha256
        || string(router_binding, "source_revision", "router inner binding")?
            != args.expected_source_revision
    {
        return Err(
            "router inner artifact binding does not match requested complete artifact".into(),
        );
    }
    let cpu_router = object(&router_inner, "cpu_oracle", "router inner")?;
    let stable_route = source_route(
        object(
            cpu_router,
            "same_capture_intermediates_retained_for_future_leased_device_parity",
            "router CPU oracle",
        )?
        .get("source_stable_route")
        .ok_or("router CPU oracle missing source stable route")?,
        "ids",
        "renormalized_weights",
        "router CPU source-stable route",
    )?;
    let expected_normalized_sha256 = string(
        object(
            object(
                cpu_router,
                "same_capture_intermediates_retained_for_future_leased_device_parity",
                "router CPU oracle",
            )?,
            "normalized_hidden",
            "router CPU oracle",
        )?,
        "sha256",
        "router CPU normalized hidden",
    )?;
    canonical_sha256(
        expected_normalized_sha256,
        "router CPU normalized-hidden SHA",
    )?;

    let admission = CompleteBinaryAdmission {
        model: QwenCompleteBinaryModel::Qwen80CoderNext,
        expected_manifest_seal_sha256: args.expected_manifest_seal_sha256.clone(),
        expected_source_audit_seal_sha256: args.expected_source_audit_seal_sha256.clone(),
        expected_source_revision: args.expected_source_revision.clone(),
    };
    // Exactly one full strict scan, then catalog re-use for all thirty
    // direct-packed wave bodies.  No source BF16/MPS model is opened.
    let catalog = Qwen80CompleteArtifactCatalog::load(&args.manifest, &admission)
        .map_err(|error| error.to_string())?;
    let hybrid = catalog
        .complete_hybrid_decoder_plan(2)
        .map_err(|error| error.to_string())?;
    let authority = Qwen80AllTenRoutedExpertPlanAuthority {
        manifest_document_sha256: &manifest_document_sha256,
        plan_document_sha256: &plan_document_sha256,
        router_receipt_sha256: &router_receipt_sha256,
        router_outer_receipt_sha256: &router_outer_receipt_sha256,
        router_outer_receipt_seal_sha256: outer_seal,
    };
    let bound_plan = hybrid
        .bind_all_ten_routed_expert_plan(0, &authority, &plan_document)
        .map_err(|error| error.to_string())?;
    let normalized_hidden = normalize_router_fixture(&catalog)?;
    let normalized_hidden_sha256 = f32_sha256(&normalized_hidden)?;
    if normalized_hidden_sha256 != expected_normalized_sha256 {
        return Err(
            "reconstructed deterministic router normalized hidden does not match sealed CPU oracle"
                .into(),
        );
    }
    let result = catalog
        .execute_all_ten_routed_expert_cpu_oracle(&bound_plan, &normalized_hidden)
        .map_err(|error| error.to_string())?;
    if result.route != stable_route {
        return Err(
            "all-ten descriptor route differs from sealed router source-stable route".into(),
        );
    }

    Ok(json!({
        "schema": RESULT_SCHEMA,
        "status": RESULT_STATUS,
        "mode": "cpu-oracle",
        "artifact_binding": {
            "manifest_path": args.manifest,
            "manifest_document_sha256": manifest_document_sha256,
            "manifest_seal_sha256": args.expected_manifest_seal_sha256,
            "source_revision": args.expected_source_revision,
            "admission_scan_performed_once_before_catalog_reuse": true,
            "all_thirty_wave_payloads_use_admission_verified_immutable_snapshots": true,
        },
        "route_plan_binding": {
            "path": args.route_plan,
            "document_sha256": plan_document_sha256,
            "router_inner_path": args.router_inner,
            "router_inner_document_sha256": router_receipt_sha256,
            "router_outer_path": args.router_outer,
            "router_outer_document_sha256": router_outer_receipt_sha256,
            "router_outer_seal_sha256": outer_seal,
            "stable_source_route_ids": result.route.ids.to_vec(),
            "stable_source_route_weights": result.route.weights.to_vec(),
        },
        "cpu_oracle": {
            "normalized_hidden_sha256": result.normalized_hidden_sha256,
            "routed_expert_sum_sha256": result.routed_expert_sum_sha256,
            "all_ten_waves_executed": result.witnesses.len() == TOP_K,
            "waves": result.witnesses.iter().map(|witness| json!({
                "wave_index": witness.wave_index,
                "expert": witness.expert,
                "normalized_weight": witness.normalized_weight,
                "gate_artifact_sha256": witness.direct_packed_gate_artifact_sha256,
                "up_artifact_sha256": witness.direct_packed_up_artifact_sha256,
                "down_artifact_sha256": witness.direct_packed_down_artifact_sha256,
                "weighted_output_sha256": witness.weighted_output_sha256,
                "gate_projection_elements": witness.oracle.gate_projection.len(),
                "up_projection_elements": witness.oracle.up_projection.len(),
                "swiglu_elements": witness.oracle.gated_up_product.len(),
                "down_output_elements": witness.oracle.output.len(),
            })).collect::<Vec<_>>(),
        },
        "future_device_graph": {
            "prepared_not_executed": true,
            "requires_new_immutable_component_only_quiet_lease": true,
            "requires_outer_reaping_durable_stdout_stderr_exit_and_receipt_last": true,
            "same_normalized_hidden_and_all_ten_cpu_witnesses_must_be_retained_in_the_device_capture": true,
            "one_command_graph_scope": "all ten gate/up/down bodies and ten source-normalized output vectors only",
            "minimum_projection_dispatches_before_fusion": TOP_K * 3,
            "does_not_include_shared_expert_route_combine_residual_or_any_token_work": true,
        },
        "claim_boundary": {
            "route65_is_precedent_only_not_coverage_for_routes_1_through_9": true,
            "no_metal_context_or_gpu_dispatch_performed": true,
            "no_shared_expert_route_combine_or_residual_performed": true,
            "no_complete_layer_token_decoder_generation_hcli_tps_tg_or_tournament_claim": true,
        },
    }))
}

fn main() {
    let args = parse_args().unwrap_or_else(|error| {
        eprintln!("Qwen80 all-ten routed-expert CPU oracle refused: {error}");
        process::exit(2);
    });
    fs::create_dir(&args.capture_dir).unwrap_or_else(|error| {
        eprintln!(
            "Qwen80 all-ten routed-expert CPU oracle refused to create exclusive capture: {error}"
        );
        process::exit(2);
    });
    let invocation = json!({
        "schema": "hawking.ascension.qwen80_all_ten_routed_expert_cpu_capture.v1",
        "pid": process::id(),
        "mode": "cpu-oracle",
        "metal_or_gpu_allowed": false,
        "manifest": args.manifest,
        "route_plan": args.route_plan,
        "router_inner": args.router_inner,
        "router_outer": args.router_outer,
    });
    write_capture(
        &args.capture_dir,
        "invocation.json",
        &serde_json::to_vec_pretty(&invocation).unwrap(),
    )
    .unwrap_or_else(|error| {
        eprintln!("Qwen80 all-ten routed-expert CPU oracle capture refused: {error}");
        process::exit(2);
    });
    match run(&args) {
        Ok(result) => {
            let stdout = serde_json::to_vec_pretty(&result).unwrap();
            write_capture(&args.capture_dir, "stdout.jsonl", &stdout).unwrap_or_else(|error| {
                eprintln!("Qwen80 all-ten routed-expert CPU oracle capture refused: {error}");
                process::exit(2);
            });
            // Receipt is deliberately last: a future outer launcher may seal
            // this complete CPU-only document, but it must not promote it to
            // Metal/layer/token evidence.
            write_capture(&args.capture_dir, "receipt.json", &stdout).unwrap_or_else(|error| {
                eprintln!("Qwen80 all-ten routed-expert CPU oracle capture refused: {error}");
                process::exit(2);
            });
        }
        Err(error) => {
            let failure = json!({
                "schema": RESULT_SCHEMA,
                "status": "REFUSED_QWEN80_ALL_TEN_ROUTED_EXPERT_CPU_ORACLE",
                "mode": "cpu-oracle",
                "error": error,
                "claim_boundary": "no execution result is earned on refusal",
            });
            let bytes = serde_json::to_vec_pretty(&failure).unwrap();
            let _ = write_capture(&args.capture_dir, "stderr.log", &bytes);
            let _ = write_capture(&args.capture_dir, "receipt.json", &bytes);
            eprintln!("Qwen80 all-ten routed-expert CPU oracle refused: {error}");
            process::exit(2);
        }
    }
}

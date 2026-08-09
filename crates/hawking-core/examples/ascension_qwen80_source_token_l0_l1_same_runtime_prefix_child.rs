//! CPU/build-only plan for one future same-runtime Qwen80 L0→L1 prefix.
//!
//! This executable deliberately does not open an artifact, construct a Metal
//! context, allocate state, issue a quiet lease, or spawn a child.  It emits
//! only the fixed command-graph authority consumed by the independent outer
//! preflight.  The preflight identity is computed by that outer from sealed
//! L0 outer/inner/assessor/admission/schedule evidence and this exact built
//! child SHA.  In particular, a historical L0 receipt is baseline provenance
//! only: it cannot transfer its process-local `PinnedBuffer` into a new
//! process.
//!
//! The future physical body is intentionally narrow and already represented
//! by `Qwen80CompleteNativeRuntime::encode_source_token_l1_deltanet_prefix_from_canonical_l0_continuation_into`:
//! rerun L0's source-token 9+14 graph, retain its live second residual in the
//! same runtime/TCB, append L1/slot-1's nine DeltaNet dispatches, fence once,
//! and record fresh L0/L1 state/output witnesses.  This plan neither executes
//! that body nor authorizes a lease.

#![recursion_limit = "256"]

use serde_json::{json, Map, Value};
use sha2::{Digest, Sha256};
use std::collections::BTreeMap;
use std::env;
use std::fs::{self, OpenOptions};
use std::io::Write;
use std::path::{Path, PathBuf};

const SCHEMA: &str =
    "hawking.ascension.qwen80_source_token_l0_l1_same_runtime_prefix_child_preflight.v1";
const STATUS: &str =
    "PREPARED_QWEN80_SOURCE_TOKEN_L0_REENCODE_AND_L1_SLOT1_PREFIX_SAME_RUNTIME_CHILD_NOT_EXECUTED";
const SOURCE_TOKEN_ID: u64 = 1;
const L0_PREFIX_DISPATCHES: u64 = 9;
const L0_SUFFIX_DISPATCHES: u64 = 14;
const L0_DISPATCHES: u64 = L0_PREFIX_DISPATCHES + L0_SUFFIX_DISPATCHES;
const L1_PREFIX_DISPATCHES: u64 = 9;
const TOTAL_DISPATCHES: u64 = L0_DISPATCHES + L1_PREFIX_DISPATCHES;

const L1_PREFIX: [(&str, &str); 9] = [
    ("input_rmsnorm", "qwen_next_direct_packed_input_rmsnorm"),
    ("qkvz_projection", "qwen_binary_sign_scale_matvec"),
    ("ba_projection", "qwen_binary_sign_scale_matvec"),
    ("qkvz_rearrange_conv", "qwen_next_qkvz_rearrange_conv_l2"),
    ("ba_decay_beta", "qwen_next_ba_to_decay_beta"),
    ("deltanet_recurrent", "qwen_next_gated_delta_decode_single"),
    ("deltanet_gated_rmsnorm", "qwen_next_deltanet_gated_rmsnorm"),
    ("out_projection", "qwen_binary_sign_scale_matvec"),
    ("first_residual", "qwen_next_add_residual"),
];

// This plan is deliberately a static authority emitter rather than the
// future physical joint host.  Preserve the complete kernel sequence anyway
// so a later host cannot satisfy the plan with only a numeric 23+9 count.
const L0_TRUE_MOE_KERNELS: [&str; 23] = [
    "qwen_next_direct_packed_input_rmsnorm",
    "qwen_binary_sign_scale_matvec",
    "qwen_binary_sign_scale_matvec",
    "qwen_next_qkvz_rearrange_conv_l2",
    "qwen_next_ba_to_decay_beta",
    "qwen_next_gated_delta_decode_single",
    "qwen_next_deltanet_gated_rmsnorm",
    "qwen_binary_sign_scale_matvec",
    "qwen_next_add_residual",
    "qwen80_postnorm_router_top10_rmsnorm",
    "qwen80_postnorm_router_top10_matvec",
    "qwen80_postnorm_router_top10_select",
    "qwen80_all_ten_routed_wave_route_guard",
    "qwen80_all_ten_routed_wave_gate_up",
    "qwen80_all_ten_routed_wave_swiglu",
    "qwen80_all_ten_routed_wave_down_weighted",
    "qwen80_shared_expert_wave_gate_up",
    "qwen80_shared_expert_wave_swiglu",
    "qwen80_shared_expert_wave_down",
    "qwen80_shared_expert_wave_scalar_gate",
    "qwen80_shared_expert_wave_apply_sigmoid_gate",
    "qwen80_moe_wave_aggregate_second_residual_route_sum",
    "qwen80_moe_wave_aggregate_second_residual_add_shared_residual",
];

#[derive(Clone, Debug)]
struct Args {
    preflight_identity_sha256: String,
    future_child_sha256: String,
    future_joint_host_binary_bound: bool,
    out: PathBuf,
}

fn sha256_hex(bytes: &[u8]) -> String {
    format!("{:x}", Sha256::digest(bytes))
}

fn is_lower_sha256(value: &str) -> bool {
    value.len() == 64
        && value
            .bytes()
            .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte))
}

fn canonical_json(value: &Value) -> Value {
    match value {
        Value::Array(values) => Value::Array(values.iter().map(canonical_json).collect()),
        Value::Object(values) => {
            let mut ordered = BTreeMap::new();
            for (key, value) in values {
                ordered.insert(key.clone(), canonical_json(value));
            }
            Value::Object(ordered.into_iter().collect::<Map<_, _>>())
        }
        value => value.clone(),
    }
}

fn sha256_json(value: &Value) -> Result<String, String> {
    serde_json::to_vec(&canonical_json(value))
        .map(|bytes| sha256_hex(&bytes))
        .map_err(|error| format!("cannot encode canonical JSON: {error}"))
}

fn seal(document: &mut Value) -> Result<String, String> {
    let object = document
        .as_object_mut()
        .ok_or("preflight document must be an object")?;
    if object.contains_key("seal_sha256") {
        return Err("preflight document must not already contain a seal".into());
    }
    let seal = sha256_json(document)?;
    document
        .as_object_mut()
        .expect("validated object")
        .insert("seal_sha256".into(), Value::String(seal.clone()));
    Ok(seal)
}

fn verify_seal(document: &Value) -> Result<String, String> {
    let object = document
        .as_object()
        .ok_or("preflight document must be an object")?;
    let observed = object
        .get("seal_sha256")
        .and_then(Value::as_str)
        .filter(|value| is_lower_sha256(value))
        .ok_or("preflight document lacks a lowercase seal_sha256")?;
    let mut unsigned = object.clone();
    unsigned.remove("seal_sha256");
    let expected = sha256_json(&Value::Object(unsigned))?;
    if observed != expected {
        return Err("preflight document seal mismatch".into());
    }
    Ok(observed.into())
}

fn exact_prefix() -> Vec<Value> {
    L1_PREFIX
        .iter()
        .enumerate()
        .map(|(index, (stage, kernel))| {
            json!({
                "ordinal": index + 1,
                "stage": stage,
                "kernel": kernel,
            })
        })
        .collect()
}

fn exact_l0_kernel_trace() -> Vec<Value> {
    L0_TRUE_MOE_KERNELS
        .iter()
        .enumerate()
        .map(|(index, kernel)| json!({"ordinal": index + 1, "kernel": kernel}))
        .collect()
}

fn exact_joint_kernel_trace() -> Vec<Value> {
    L0_TRUE_MOE_KERNELS
        .iter()
        .chain(L1_PREFIX.iter().map(|(_, kernel)| kernel))
        .enumerate()
        .map(|(index, kernel)| json!({"ordinal": index + 1, "kernel": kernel}))
        .collect()
}

fn prepared_document(args: &Args) -> Value {
    json!({
        "schema": SCHEMA,
        "status": STATUS,
        "future_joint_l0_l1_child_sha256": args.future_child_sha256,
        "preflight_identity_sha256": args.preflight_identity_sha256,
        "child_started": false,
        "metal_or_gpu_activity_performed": false,
        "component_only": true,
        "same_runtime_required": true,
        "same_session_required": true,
        "same_tcb_required": true,
        "baseline_l0_receipts_provenance_only": true,
        "cross_process_pinned_buffer_transfer_allowed": false,
        "external_l0_buffer_or_state_import_allowed": false,
        "opaque_canonical_l0_continuation_required": true,
        "raw_pinned_buffer_or_dispatch_count_input_allowed": false,
        "opaque_capability_must_bind_runtime_state_arena_identity": true,
        "future_joint_host_binary_bound": args.future_joint_host_binary_bound,
        "future_joint_host_binary_role": if args.future_joint_host_binary_bound { "strict_joint_l0_l1_same_runtime_host" } else { "static_preflight_authority_only_not_joint_host" },
        "l0_reencode": {
            "source_token_id": SOURCE_TOKEN_ID,
            "prefix_dispatches": L0_PREFIX_DISPATCHES,
            "suffix_dispatches": L0_SUFFIX_DISPATCHES,
            "total_dispatches": L0_DISPATCHES,
            "same_tcb_fence_required": true,
            "historical_l0_receipt_may_only_supply_baseline_parity_and_provenance": true,
        },
        "l1_prefix": {
            "layer": 1,
            "mixer": "delta_net",
            "linear_state_slot": 1,
            "prefix_dispatches": L1_PREFIX_DISPATCHES,
            "exact_prefix_dispatches": exact_prefix(),
            "no_l1_suffix_or_moe_dispatch_authorized": true,
            "retained_l0_output_must_be_owned_by_the_opaque_canonical_l0_continuation": true,
            "l1_active_and_rollback_state_witnesses_required_after_fence": true,
            "l1_output_elements": 2048,
            "l1_output_bytes": 8192,
        },
        "joint_command_graph": {
            "l0_dispatches": L0_DISPATCHES,
            "l1_prefix_dispatches": L1_PREFIX_DISPATCHES,
            "total_dispatches": TOTAL_DISPATCHES,
            "same_runtime_same_tcb_required": true,
            "same_session_required": true,
            "single_fence_after_l0_and_l1_prefix_required": true,
            "non_timed_token_command_buffer_required": true,
            "tcb_trace_mode": "off",
            "runtime_api": "Qwen80CompleteNativeRuntime::encode_source_token_l1_deltanet_prefix_from_canonical_l0_continuation_into",
            "opaque_capability_factory": "Qwen80CompleteNativeRuntime::certify_source_token_l0_true_moe_continuation",
            "consuming_finalizer": "Qwen80SameRuntimeLayer1DeltaNetPrefixEncoder::finalize_after_exact_joint_fence",
            "runtime_api_requires_exact_preceding_l0_dispatches": L0_DISPATCHES,
            "runtime_api_requires_exact_total_dispatches_after_l1": TOTAL_DISPATCHES,
            "structural_kernel_trace_required": true,
            "exact_l0_kernel_trace": exact_l0_kernel_trace(),
            "exact_joint_kernel_trace": exact_joint_kernel_trace(),
            "finalizer_must_consume_the_only_command_buffer_before_fence": true,
            "fresh_l0_suffix_readbacks_required": {
                "route_guard": true,
                "postnorm": true,
                "router_logits": true,
                "all_ten_weighted_route_witnesses": 10,
                "shared_output": true,
                "routed_sum": true,
                "second_residual": true,
            },
            "fresh_l0_and_l1_state_output_rollback_witnesses_required": true,
        },
        "claim_boundary": {
            "complete_layer_or_token_allowed": false,
            "decoder_generation_hcli_tps_tg_or_tournament_allowed": false,
            "l1_suffix_or_moe_allowed": false,
            "automatic_retry_allowed": false,
            "lease_issued_or_consumed": false,
            "watcher_server_or_runtime_transition_authorized": false,
        },
    })
}

fn canonical_new_file(path: &Path, label: &str) -> Result<PathBuf, String> {
    if !path.is_absolute() {
        return Err(format!("{label} must be an absolute path"));
    }
    let parent = path
        .parent()
        .filter(|parent| parent.is_absolute())
        .ok_or_else(|| format!("{label} needs an absolute parent"))?;
    let metadata = fs::symlink_metadata(parent)
        .map_err(|error| format!("cannot stat {label} parent {}: {error}", parent.display()))?;
    if metadata.file_type().is_symlink() || !metadata.is_dir() {
        return Err(format!("{label} parent must be a real directory"));
    }
    if path.exists() {
        return Err(format!("{label} must be create-new and does not overwrite"));
    }
    Ok(path.to_path_buf())
}

fn canonical_regular(path: &Path, label: &str) -> Result<PathBuf, String> {
    if !path.is_absolute() {
        return Err(format!("{label} must be an absolute path"));
    }
    let metadata = fs::symlink_metadata(path)
        .map_err(|error| format!("cannot stat {label} {}: {error}", path.display()))?;
    if metadata.file_type().is_symlink() || !metadata.is_file() {
        return Err(format!("{label} must be a regular non-symlink file"));
    }
    fs::canonicalize(path).map_err(|error| format!("cannot canonicalize {label}: {error}"))
}

fn sha256_file(path: &Path, label: &str) -> Result<String, String> {
    let bytes = fs::read(path)
        .map_err(|error| format!("cannot read {label} {}: {error}", path.display()))?;
    Ok(sha256_hex(&bytes))
}

fn write_new(path: &Path, bytes: &[u8]) -> Result<(), String> {
    let mut file = OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(path)
        .map_err(|error| format!("cannot create {}: {error}", path.display()))?;
    file.write_all(bytes)
        .map_err(|error| format!("cannot write {}: {error}", path.display()))?;
    file.sync_all()
        .map_err(|error| format!("cannot sync {}: {error}", path.display()))
}

fn parse_args(arguments: &[String]) -> Result<Args, String> {
    let mut preflight_identity_sha256 = None;
    let mut future_child_sha256 = None;
    let mut future_host_bin = None;
    let mut out = None;
    let mut index = 1usize;
    while index < arguments.len() {
        let value = arguments
            .get(index + 1)
            .ok_or_else(|| format!("{} needs a value", arguments[index]))?;
        match arguments[index].as_str() {
            "--preflight-identity" => preflight_identity_sha256 = Some(value.clone()),
            "--future-child-sha" => future_child_sha256 = Some(value.clone()),
            "--future-host-bin" => future_host_bin = Some(PathBuf::from(value)),
            "--out" => out = Some(PathBuf::from(value)),
            flag => return Err(format!("unknown argument {flag:?}")),
        }
        index += 2;
    }
    let preflight_identity_sha256 =
        preflight_identity_sha256.ok_or("--preflight-identity SHA256 is required")?;
    let future_joint_host_binary_bound = future_host_bin.is_some();
    let future_child_sha256 = if let Some(host_bin) = future_host_bin {
        let host_bin = canonical_regular(&host_bin, "--future-host-bin")?;
        let observed = sha256_file(&host_bin, "--future-host-bin")?;
        if let Some(claimed) = future_child_sha256 {
            if claimed != observed {
                return Err("--future-child-sha does not match --future-host-bin bytes".into());
            }
        }
        observed
    } else {
        future_child_sha256.ok_or(
            "--future-child-sha SHA256 is required unless --future-host-bin binds a concrete host",
        )?
    };
    if !is_lower_sha256(&preflight_identity_sha256) {
        return Err("--preflight-identity must be a lowercase SHA-256".into());
    }
    if !is_lower_sha256(&future_child_sha256) {
        return Err("--future-child-sha must be a lowercase SHA-256".into());
    }
    let out = canonical_new_file(&out.ok_or("--out ABSOLUTE_NEW_PATH is required")?, "--out")?;
    Ok(Args {
        preflight_identity_sha256,
        future_child_sha256,
        future_joint_host_binary_bound,
        out,
    })
}

fn run(args: Args) -> Result<String, String> {
    let mut document = prepared_document(&args);
    let seal = seal(&mut document)?;
    verify_seal(&document)?;
    let bytes = serde_json::to_vec_pretty(&document)
        .map_err(|error| format!("cannot encode preflight: {error}"))?;
    write_new(&args.out, &bytes)?;
    Ok(seal)
}

fn usage() -> &'static str {
    "usage: ascension_qwen80_source_token_l0_l1_same_runtime_prefix_child \\\n+--preflight-identity SHA256 --future-child-sha SHA256 --out ABSOLUTE_NEW_PATH"
}

fn main() {
    match parse_args(&env::args().collect::<Vec<_>>()).and_then(run) {
        Ok(seal) => println!("prepared sealed joint L0→L1 CPU-only child preflight: {seal}"),
        Err(error) => {
            eprintln!("{error}\n{}", usage());
            std::process::exit(2);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sha(byte: char) -> String {
        byte.to_string().repeat(64)
    }

    #[test]
    fn prepared_plan_is_sealed_and_locks_the_joint_23_plus_9_graph() {
        let args = Args {
            preflight_identity_sha256: sha('a'),
            future_child_sha256: sha('b'),
            future_joint_host_binary_bound: false,
            out: PathBuf::from("/unused"),
        };
        let mut document = prepared_document(&args);
        assert_eq!(document["schema"], SCHEMA);
        assert_eq!(document["status"], STATUS);
        assert_eq!(
            document["joint_command_graph"]["l0_dispatches"],
            L0_DISPATCHES
        );
        assert_eq!(
            document["joint_command_graph"]["l1_prefix_dispatches"],
            L1_PREFIX_DISPATCHES
        );
        assert_eq!(
            document["joint_command_graph"]["total_dispatches"],
            TOTAL_DISPATCHES
        );
        assert_eq!(
            document["joint_command_graph"]["exact_l0_kernel_trace"]
                .as_array()
                .unwrap()
                .len(),
            L0_DISPATCHES as usize
        );
        assert_eq!(
            document["joint_command_graph"]["exact_joint_kernel_trace"]
                .as_array()
                .unwrap()
                .len(),
            TOTAL_DISPATCHES as usize
        );
        assert_eq!(
            document["l1_prefix"]["exact_prefix_dispatches"]
                .as_array()
                .unwrap()
                .len(),
            9
        );
        assert_eq!(
            document["cross_process_pinned_buffer_transfer_allowed"],
            false
        );
        assert_eq!(document["opaque_canonical_l0_continuation_required"], true);
        assert_eq!(
            document["opaque_capability_must_bind_runtime_state_arena_identity"],
            true
        );
        assert_eq!(document["same_session_required"], true);
        assert_eq!(document["future_joint_host_binary_bound"], false);
        assert_eq!(
            document["joint_command_graph"]["tcb_trace_mode"],
            "off"
        );
        assert_eq!(
            document["joint_command_graph"]
                ["finalizer_must_consume_the_only_command_buffer_before_fence"],
            true
        );
        assert_eq!(
            document["joint_command_graph"]["fresh_l0_suffix_readbacks_required"]
                ["all_ten_weighted_route_witnesses"],
            10
        );
        seal(&mut document).unwrap();
        assert!(verify_seal(&document).is_ok());
    }

    #[test]
    fn plan_explicitly_refuses_import_and_promotion() {
        let args = Args {
            preflight_identity_sha256: sha('c'),
            future_child_sha256: sha('d'),
            future_joint_host_binary_bound: false,
            out: PathBuf::from("/unused"),
        };
        let document = prepared_document(&args);
        assert_eq!(
            document["external_l0_buffer_or_state_import_allowed"],
            false
        );
        assert_eq!(
            document["raw_pinned_buffer_or_dispatch_count_input_allowed"],
            false
        );
        assert_eq!(
            document["claim_boundary"]["complete_layer_or_token_allowed"],
            false
        );
        assert_eq!(
            document["claim_boundary"]["l1_suffix_or_moe_allowed"],
            false
        );
        assert_eq!(
            document["claim_boundary"]["lease_issued_or_consumed"],
            false
        );
    }

    #[test]
    fn parser_refuses_non_sha_values_and_non_absolute_outputs() {
        let error = parse_args(&[
            "child".into(),
            "--preflight-identity".into(),
            "not-a-sha".into(),
            "--future-child-sha".into(),
            sha('a'),
            "--out".into(),
            "/tmp/unused".into(),
        ])
        .unwrap_err();
        assert!(error.contains("preflight-identity"));
        let error = parse_args(&[
            "child".into(),
            "--preflight-identity".into(),
            sha('a'),
            "--future-child-sha".into(),
            sha('b'),
            "--out".into(),
            "relative.json".into(),
        ])
        .unwrap_err();
        assert!(error.contains("absolute"));
    }
}

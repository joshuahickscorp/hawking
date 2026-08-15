//! Static Qwen3-Coder-Next 48-layer payload and schedule authority.
//!
//! This CPU-only program turns two already-sealed JSON authorities into one
//! immutable, create-new *assembly input* for a future Qwen80 command graph:
//!
//! - the complete direct-packed descriptor inventory; and
//! - the sealed source-config metadata authority.
//!
//! It reads descriptor metadata only.  It never opens an artifact payload or a
//! source shard, creates a Metal context, dispatches a kernel, starts a runtime
//! or server, contacts HCLI, executes a token, or measures TPS/TG.  In
//! particular, its `PREPARED_NOT_EXECUTED` result is not decoder readiness.
//!
//! The output contains every one of the 74,391 source tensor descriptors in
//! future command order, including every expert projection.  A later graph can
//! therefore bind a raw plan SHA plus the sealed inventory SHA without
//! rediscovering a tensor name or silently changing the 3x DeltaNet / 1x GQA
//! schedule.
//!
//! ```text
//! cargo run -p hawking-core --example ascension_qwen80_48_layer_payload_schedule_authority -- \
//!   --descriptor-inventory /absolute/path/QWEN80_COMPLETE_BINARY_GRAVITY_CANDIDATE.json \
//!   --source-config-authority /absolute/path/QWEN80_SOURCE_METADATA_CANDIDATE.json \
//!   --out /absolute/new/QWEN80_48_LAYER_PAYLOAD_SCHEDULE_AUTHORITY.json
//! ```

use serde::{Deserialize, Serialize};
use serde_json::Value;
use sha2::{Digest, Sha256};
use std::collections::{BTreeMap, BTreeSet};
use std::env;
use std::fs::{self, OpenOptions};
use std::io::Write;
use std::path::{Path, PathBuf};

const SCHEMA: &str = "hawking.ascension.qwen80_48_layer_payload_schedule_authority.v1";
const STATUS: &str = "PREPARED_QWEN80_48_LAYER_PAYLOAD_SCHEDULE_AUTHORITY_NOT_EXECUTED";
const EXECUTION_STATUS: &str = "PREPARED_NOT_EXECUTED";

const MANIFEST_SCHEMA: &str = "hawking.ascension.qwen80_complete_binary_gravity.v1";
const MANIFEST_STATUS: &str = "CANDIDATE_COMPLETE_BINARY_ARTIFACT_LOW_FIDELITY_UNQUALIFIED";
const SOURCE_CONFIG_SCHEMA: &str = "hawking.ascension.source_admission_candidate.v1";
const SOURCE_CONFIG_STATUS: &str = "CANDIDATE_METADATA_CAPTURED";

const MODEL_ID: &str = "Qwen3-Coder-Next-80B";
const MODEL_KEY: &str = "qwen80";
const SOURCE_REPOSITORY: &str = "Qwen/Qwen3-Coder-Next";
const SOURCE_REVISION: &str = "a7fbcb5c0e12d62a448eaa0e260346bf5dcc0feb";
const SOURCE_CONFIG_SHA256: &str =
    "a7b8098d3b05777f12bb5677a26bf1240a1bb09def1b06b29e6be86cae2e84f8";

const LAYERS: usize = 48;
const DELTANET_LAYERS: usize = 36;
const GQA_LAYERS: usize = 12;
const HIDDEN: usize = 2_048;
const FULL_ATTN_HEADS: usize = 16;
const FULL_ATTN_KV_HEADS: usize = 2;
const FULL_ATTN_HEAD_DIM: usize = 256;
const LINEAR_KEY_HEADS: usize = 16;
const LINEAR_VALUE_HEADS: usize = 32;
const LINEAR_KEY_HEAD_DIM: usize = 128;
const LINEAR_VALUE_HEAD_DIM: usize = 128;
const LINEAR_CONV_KERNEL: usize = 4;
const EXPERTS: usize = 512;
const TOP_K: usize = 10;
const INTERMEDIATE: usize = 512;
const VOCAB: usize = 151_936;
const TOKENIZER_VOCAB: usize = 151_669;
const TAIL_ROWS: usize = VOCAB - TOKENIZER_VOCAB;
const GROUP_SIZE: usize = 128;
const COMPLETE_TENSOR_COUNT: usize = 74_391;

#[derive(Clone, Debug)]
struct Args {
    descriptor_inventory: PathBuf,
    source_config_authority: PathBuf,
    out: PathBuf,
}

#[derive(Clone, Debug, Deserialize)]
struct ManifestSource {
    repository: String,
    tensor_count: usize,
}

#[derive(Clone, Debug, Deserialize)]
struct ManifestInventory {
    schema: String,
    status: String,
    seal_sha256: String,
    source: ManifestSource,
    tensors: Vec<TensorDescriptor>,
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
struct PackedLayout {
    magic: String,
    group_size: usize,
    scale_dtype: String,
    sign_bit_order: String,
    version: u32,
}

#[derive(Clone, Debug, Deserialize)]
struct TensorDescriptor {
    tensor_name: String,
    shape: Vec<usize>,
    elements: usize,
    artifact_path: String,
    artifact_bytes: u64,
    artifact_sha256: String,
    source_dtype: String,
    source_shard: String,
    source_shard_sha256: String,
    layout: PackedLayout,
}

#[derive(Clone, Debug, Deserialize)]
struct SourceConfigSource {
    repository: String,
    revision: String,
}

#[derive(Clone, Debug, Deserialize)]
struct SourceArchitecture {
    architectures: Vec<String>,
    config_captured: bool,
    config_sha256: String,
    hidden_size: usize,
    model_type: String,
    num_experts: usize,
    num_experts_per_tok: usize,
    num_hidden_layers: usize,
    vocab_size: usize,
}

#[derive(Clone, Debug, Deserialize)]
struct SourceConfigAuthority {
    schema: String,
    status: String,
    seal_sha256: String,
    source: SourceConfigSource,
    architecture: SourceArchitecture,
}

#[derive(Clone, Copy, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(rename_all = "snake_case")]
enum Mixer {
    DeltaNet,
    Gqa,
}

impl Mixer {
    fn expected(layer: usize) -> Self {
        if layer % 4 == 3 {
            Self::Gqa
        } else {
            Self::DeltaNet
        }
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize)]
#[serde(rename_all = "snake_case")]
enum StateDomain {
    DeltaNetConvAndRecurrent,
    GqaKv,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize)]
struct Geometry {
    layer_count: usize,
    hidden_size: usize,
    full_attention_heads: usize,
    full_attention_key_value_heads: usize,
    full_attention_head_dim: usize,
    linear_key_heads: usize,
    linear_value_heads: usize,
    linear_key_head_dim: usize,
    linear_value_head_dim: usize,
    linear_conv_kernel: usize,
    experts: usize,
    top_k: usize,
    moe_intermediate: usize,
    shared_expert_intermediate: usize,
    vocab_size: usize,
    tokenizer_vocab_size: usize,
    reserved_lm_head_tail_rows: usize,
    direct_pack_group_size: usize,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize)]
struct SourceAuthorityBinding {
    model_id: &'static str,
    model_key: &'static str,
    source_repository: &'static str,
    source_revision: &'static str,
    source_config_sha256: &'static str,
    descriptor_inventory_canonical_path: String,
    descriptor_inventory_document_sha256: String,
    descriptor_inventory_schema: String,
    descriptor_inventory_status: String,
    descriptor_inventory_seal_sha256: String,
    descriptor_inventory_tensor_count: usize,
    source_config_authority_canonical_path: String,
    source_config_authority_document_sha256: String,
    source_config_authority_schema: String,
    source_config_authority_status: String,
    source_config_authority_seal_sha256: String,
}

/// A complete copy of the sealed descriptor metadata required to address one
/// direct-packed tensor.  The descriptor inventory document SHA and this
/// per-tensor identity both have to match before a future graph may upload it.
#[derive(Clone, Debug, Eq, PartialEq, Serialize)]
struct TensorBinding {
    inventory_ordinal: usize,
    role: String,
    tensor_name: String,
    shape: Vec<usize>,
    elements: usize,
    artifact_path: String,
    artifact_bytes: u64,
    artifact_sha256: String,
    source_dtype: String,
    source_shard: String,
    source_shard_sha256: String,
    layout: PackedLayout,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize)]
struct StateSlot {
    layer: usize,
    slot: usize,
    domain: StateDomain,
    device_buffers_required_before_execution: Vec<&'static str>,
    rollback_buffers_required_before_execution: Vec<&'static str>,
    state_materialized_by_this_plan: bool,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize)]
struct RoutedExpertProjectionTable {
    projection: &'static str,
    expected_shape: Vec<usize>,
    source_expert_order: Vec<usize>,
    descriptors: Vec<TensorBinding>,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize)]
struct RoutedExpertPayloads {
    expert_count: usize,
    top_k: usize,
    projection_execution_order: [&'static str; 3],
    tables: Vec<RoutedExpertProjectionTable>,
    route_selection_materialized_by_this_plan: bool,
    expert_payload_opened_by_this_plan: bool,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize)]
struct SharedExpertPayloads {
    execution_order: [&'static str; 4],
    gate_proj: TensorBinding,
    up_proj: TensorBinding,
    down_proj: TensorBinding,
    scalar_gate: TensorBinding,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize)]
struct LayerPayloadPlan {
    layer: usize,
    mixer: Mixer,
    state_slot: StateSlot,
    input_layernorm: TensorBinding,
    mixer_tensor_execution_order: Vec<&'static str>,
    mixer_tensors: Vec<TensorBinding>,
    post_attention_layernorm: TensorBinding,
    router_gate: TensorBinding,
    routed_experts: RoutedExpertPayloads,
    shared_expert: SharedExpertPayloads,
    layer_command_boundary_order: Vec<&'static str>,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize)]
struct TerminalHeadPlan {
    final_norm: TensorBinding,
    lm_head: TensorBinding,
    all_row_lm_head_rows: usize,
    tokenizer_addressable_rows: usize,
    reserved_tail_rows: usize,
    execution_order: [&'static str; 5],
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize)]
struct GraphStateBoundary {
    ordinal: usize,
    name: &'static str,
    producer: &'static str,
    consumer: &'static str,
    shape: Vec<usize>,
    required_for_future_graph: bool,
    materialized_by_this_plan: bool,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize)]
struct ClaimBoundary {
    assembly_authority_only: bool,
    decoder_readiness_report: bool,
    artifact_payload_open_or_scan_performed: bool,
    metal_device_or_dispatch_performed: bool,
    runtime_watcher_registry_server_or_hcli_changed: bool,
    model_execution_performed: bool,
    token_generation_or_feedback_performed: bool,
    tps_or_tg_measured: bool,
    execution_status: &'static str,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize)]
struct PayloadScheduleAuthority {
    schema: &'static str,
    status: &'static str,
    source_authority: SourceAuthorityBinding,
    geometry: Geometry,
    embedding: TensorBinding,
    layers: Vec<LayerPayloadPlan>,
    deltanet_state_slots: Vec<StateSlot>,
    gqa_state_slots: Vec<StateSlot>,
    terminal_head: TerminalHeadPlan,
    future_graph_state_boundaries: Vec<GraphStateBoundary>,
    full_command_graph_order: Vec<String>,
    resolved_tensor_binding_count: usize,
    all_48_layers_scheduled: bool,
    all_descriptors_source_artifact_bound: bool,
    claim_boundary: ClaimBoundary,
}

#[derive(Clone, Debug)]
struct SealedDocument {
    canonical_path: PathBuf,
    document_sha256: String,
    value: Value,
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

fn checked_elements(shape: &[usize], label: &str) -> Result<usize, String> {
    if shape.is_empty() || shape.contains(&0) {
        return Err(format!("{label} has an empty or zero dimension"));
    }
    shape.iter().try_fold(1usize, |total, dimension| {
        total
            .checked_mul(*dimension)
            .ok_or_else(|| format!("{label} element count overflows usize"))
    })
}

fn geometry() -> Geometry {
    Geometry {
        layer_count: LAYERS,
        hidden_size: HIDDEN,
        full_attention_heads: FULL_ATTN_HEADS,
        full_attention_key_value_heads: FULL_ATTN_KV_HEADS,
        full_attention_head_dim: FULL_ATTN_HEAD_DIM,
        linear_key_heads: LINEAR_KEY_HEADS,
        linear_value_heads: LINEAR_VALUE_HEADS,
        linear_key_head_dim: LINEAR_KEY_HEAD_DIM,
        linear_value_head_dim: LINEAR_VALUE_HEAD_DIM,
        linear_conv_kernel: LINEAR_CONV_KERNEL,
        experts: EXPERTS,
        top_k: TOP_K,
        moe_intermediate: INTERMEDIATE,
        shared_expert_intermediate: INTERMEDIATE,
        vocab_size: VOCAB,
        tokenizer_vocab_size: TOKENIZER_VOCAB,
        reserved_lm_head_tail_rows: TAIL_ROWS,
        direct_pack_group_size: GROUP_SIZE,
    }
}

fn expected_tensor_count() -> usize {
    // embedding + terminal; per layer: two norms, router, four shared,
    // mixer (7 DeltaNet or 6 GQA), and 512 * gate/up/down routed experts.
    3 + LAYERS * 7 + DELTANET_LAYERS * 7 + GQA_LAYERS * 6 + LAYERS * EXPERTS * 3
}

fn expect_exact_config(config: &SourceConfigAuthority) -> Result<(), String> {
    if config.schema != SOURCE_CONFIG_SCHEMA || config.status != SOURCE_CONFIG_STATUS {
        return Err("source-config authority schema/status drifted".into());
    }
    if config.source.repository != SOURCE_REPOSITORY || config.source.revision != SOURCE_REVISION {
        return Err("source-config authority repository/revision drifted".into());
    }
    let architecture = &config.architecture;
    if !architecture.config_captured
        || architecture.config_sha256 != SOURCE_CONFIG_SHA256
        || architecture.hidden_size != HIDDEN
        || architecture.model_type != "qwen3_next"
        || architecture.num_experts != EXPERTS
        || architecture.num_experts_per_tok != TOP_K
        || architecture.num_hidden_layers != LAYERS
        || architecture.vocab_size != VOCAB
        || !architecture
            .architectures
            .iter()
            .any(|architecture| architecture == "Qwen3NextForCausalLM")
    {
        return Err("source-config authority does not describe the exact Qwen80 geometry".into());
    }
    if !is_lower_sha256(&config.seal_sha256) {
        return Err("source-config authority seal is malformed".into());
    }
    Ok(())
}

fn verify_document_seal(value: &Value, label: &str) -> Result<String, String> {
    let mut preimage = value.clone();
    let Value::Object(object) = &mut preimage else {
        return Err(format!("{label} must be a JSON object"));
    };
    let Some(seal) = object
        .remove("seal_sha256")
        .and_then(|value| value.as_str().map(str::to_owned))
    else {
        return Err(format!("{label} is missing seal_sha256"));
    };
    if !is_lower_sha256(&seal) {
        return Err(format!("{label} has a malformed seal_sha256"));
    }
    let canonical = serde_json::to_vec(&preimage)
        .map_err(|error| format!("cannot canonicalize {label}: {error}"))?;
    if sha256_hex(&canonical) != seal {
        return Err(format!("{label} seal_sha256 does not match canonical JSON"));
    }
    Ok(seal)
}

fn read_sealed_document(path: &Path, label: &str) -> Result<SealedDocument, String> {
    if !path.is_absolute() {
        return Err(format!("{label} path must be absolute"));
    }
    let canonical_path = fs::canonicalize(path)
        .map_err(|error| format!("cannot canonicalize {label} {}: {error}", path.display()))?;
    if !canonical_path.is_file() {
        return Err(format!(
            "{label} {} is not a regular file",
            canonical_path.display()
        ));
    }
    let bytes = fs::read(&canonical_path)
        .map_err(|error| format!("cannot read {label} {}: {error}", canonical_path.display()))?;
    let value: Value = serde_json::from_slice(&bytes)
        .map_err(|error| format!("cannot parse {label} JSON: {error}"))?;
    verify_document_seal(&value, label)?;
    Ok(SealedDocument {
        canonical_path,
        document_sha256: sha256_hex(&bytes),
        value,
    })
}

fn parse_args(arguments: &[String]) -> Result<Args, String> {
    let mut descriptor_inventory = None;
    let mut source_config_authority = None;
    let mut out = None;
    let mut index = 0usize;
    while index < arguments.len() {
        let flag = &arguments[index];
        let value = arguments
            .get(index + 1)
            .ok_or_else(|| format!("{flag} requires an absolute path"))?;
        let path = PathBuf::from(value);
        if !path.is_absolute() {
            return Err(format!("{flag} must name an absolute path"));
        }
        match flag.as_str() {
            "--descriptor-inventory" => {
                if descriptor_inventory.replace(path).is_some() {
                    return Err("--descriptor-inventory was supplied more than once".into());
                }
            }
            "--source-config-authority" => {
                if source_config_authority.replace(path).is_some() {
                    return Err("--source-config-authority was supplied more than once".into());
                }
            }
            "--out" => {
                if out.replace(path).is_some() {
                    return Err("--out was supplied more than once".into());
                }
            }
            _ => {
                return Err(format!(
                    "unknown argument {flag}; usage: ascension_qwen80_48_layer_payload_schedule_authority --descriptor-inventory ABSOLUTE_SEALED_JSON --source-config-authority ABSOLUTE_SEALED_JSON --out ABSOLUTE_NEW_FILE"
                ));
            }
        }
        index += 2;
    }
    let descriptor_inventory = descriptor_inventory.ok_or("missing --descriptor-inventory")?;
    let source_config_authority =
        source_config_authority.ok_or("missing --source-config-authority")?;
    let out = out.ok_or("missing --out")?;
    let Some(parent) = out.parent() else {
        return Err("--out path has no parent directory".into());
    };
    if !parent.is_dir() {
        return Err("--out parent directory must already exist".into());
    }
    if out.exists() {
        return Err("--out refuses to overwrite an existing file".into());
    }
    Ok(Args {
        descriptor_inventory,
        source_config_authority,
        out,
    })
}

fn validate_inventory(inventory: &ManifestInventory) -> Result<(), String> {
    if inventory.schema != MANIFEST_SCHEMA || inventory.status != MANIFEST_STATUS {
        return Err("descriptor inventory schema/status drifted".into());
    }
    if inventory.source.repository != SOURCE_REPOSITORY
        || inventory.source.tensor_count != COMPLETE_TENSOR_COUNT
        || inventory.tensors.len() != COMPLETE_TENSOR_COUNT
    {
        return Err("descriptor inventory source identity or tensor count drifted".into());
    }
    if !is_lower_sha256(&inventory.seal_sha256) {
        return Err("descriptor inventory seal is malformed".into());
    }
    Ok(())
}

fn validate_descriptor(
    descriptor: &TensorDescriptor,
    expected_shape: &[usize],
    label: &str,
) -> Result<(), String> {
    if descriptor.shape != expected_shape {
        return Err(format!(
            "{label} shape {:?} differs from expected {:?}",
            descriptor.shape, expected_shape
        ));
    }
    if descriptor.elements != checked_elements(expected_shape, label)? {
        return Err(format!("{label} element count disagrees with its shape"));
    }
    if descriptor.artifact_path.is_empty()
        || descriptor.artifact_bytes == 0
        || descriptor.source_dtype != "BF16"
        || descriptor.source_shard.is_empty()
        || !is_lower_sha256(&descriptor.artifact_sha256)
        || !is_lower_sha256(&descriptor.source_shard_sha256)
    {
        return Err(format!(
            "{label} descriptor identity or compact payload metadata drifted"
        ));
    }
    if descriptor.layout
        != (PackedLayout {
            magic: "HQ30G1B1".into(),
            group_size: GROUP_SIZE,
            scale_dtype: "float16".into(),
            sign_bit_order: "little".into(),
            version: 1,
        })
    {
        return Err(format!(
            "{label} direct-pack layout is not the admitted group-128 layout"
        ));
    }
    Ok(())
}

fn bind_tensor(
    index: &BTreeMap<&str, (usize, &TensorDescriptor)>,
    used: &mut BTreeSet<String>,
    role: impl Into<String>,
    name: String,
    expected_shape: &[usize],
) -> Result<TensorBinding, String> {
    let Some((inventory_ordinal, descriptor)) = index.get(name.as_str()) else {
        return Err(format!(
            "required descriptor {name} is absent from the sealed inventory"
        ));
    };
    validate_descriptor(descriptor, expected_shape, &name)?;
    if !used.insert(name.clone()) {
        return Err(format!("descriptor {name} was scheduled more than once"));
    }
    Ok(TensorBinding {
        inventory_ordinal: *inventory_ordinal,
        role: role.into(),
        tensor_name: descriptor.tensor_name.clone(),
        shape: descriptor.shape.clone(),
        elements: descriptor.elements,
        artifact_path: descriptor.artifact_path.clone(),
        artifact_bytes: descriptor.artifact_bytes,
        artifact_sha256: descriptor.artifact_sha256.clone(),
        source_dtype: descriptor.source_dtype.clone(),
        source_shard: descriptor.source_shard.clone(),
        source_shard_sha256: descriptor.source_shard_sha256.clone(),
        layout: descriptor.layout.clone(),
    })
}

fn deltanet_tensor_specifications(layer: usize) -> Vec<(&'static str, String, Vec<usize>)> {
    let prefix = format!("model.layers.{layer}.linear_attn");
    let key_dim = LINEAR_KEY_HEADS * LINEAR_KEY_HEAD_DIM;
    let value_dim = LINEAR_VALUE_HEADS * LINEAR_VALUE_HEAD_DIM;
    let conv_dim = key_dim * 2 + value_dim;
    vec![
        (
            "deltanet_in_proj_qkvz",
            format!("{prefix}.in_proj_qkvz.weight"),
            vec![key_dim * 2 + value_dim * 2, HIDDEN],
        ),
        (
            "deltanet_in_proj_ba",
            format!("{prefix}.in_proj_ba.weight"),
            vec![LINEAR_VALUE_HEADS * 2, HIDDEN],
        ),
        (
            "deltanet_causal_conv1d",
            format!("{prefix}.conv1d.weight"),
            vec![conv_dim, 1, LINEAR_CONV_KERNEL],
        ),
        (
            "deltanet_a_log",
            format!("{prefix}.A_log"),
            vec![LINEAR_VALUE_HEADS],
        ),
        (
            "deltanet_dt_bias",
            format!("{prefix}.dt_bias"),
            vec![LINEAR_VALUE_HEADS],
        ),
        (
            "deltanet_gated_rmsnorm",
            format!("{prefix}.norm.weight"),
            vec![LINEAR_VALUE_HEAD_DIM],
        ),
        (
            "deltanet_out_proj",
            format!("{prefix}.out_proj.weight"),
            vec![HIDDEN, value_dim],
        ),
    ]
}

fn gqa_tensor_specifications(layer: usize) -> Vec<(&'static str, String, Vec<usize>)> {
    let prefix = format!("model.layers.{layer}.self_attn");
    let query_dim = FULL_ATTN_HEADS * FULL_ATTN_HEAD_DIM;
    let kv_dim = FULL_ATTN_KV_HEADS * FULL_ATTN_HEAD_DIM;
    vec![
        (
            "gqa_q_proj",
            format!("{prefix}.q_proj.weight"),
            vec![query_dim * 2, HIDDEN],
        ),
        (
            "gqa_k_proj",
            format!("{prefix}.k_proj.weight"),
            vec![kv_dim, HIDDEN],
        ),
        (
            "gqa_v_proj",
            format!("{prefix}.v_proj.weight"),
            vec![kv_dim, HIDDEN],
        ),
        (
            "gqa_q_norm",
            format!("{prefix}.q_norm.weight"),
            vec![FULL_ATTN_HEAD_DIM],
        ),
        (
            "gqa_k_norm",
            format!("{prefix}.k_norm.weight"),
            vec![FULL_ATTN_HEAD_DIM],
        ),
        (
            "gqa_o_proj",
            format!("{prefix}.o_proj.weight"),
            vec![HIDDEN, query_dim],
        ),
    ]
}

fn expert_projection_table(
    layer: usize,
    projection: &'static str,
    expected_shape: Vec<usize>,
    index: &BTreeMap<&str, (usize, &TensorDescriptor)>,
    used: &mut BTreeSet<String>,
) -> Result<RoutedExpertProjectionTable, String> {
    let mut descriptors = Vec::with_capacity(EXPERTS);
    let mut source_expert_order = Vec::with_capacity(EXPERTS);
    for expert in 0..EXPERTS {
        let name = format!("model.layers.{layer}.mlp.experts.{expert}.{projection}.weight");
        descriptors.push(bind_tensor(
            index,
            used,
            format!("routed_expert_{projection}"),
            name,
            &expected_shape,
        )?);
        source_expert_order.push(expert);
    }
    Ok(RoutedExpertProjectionTable {
        projection,
        expected_shape,
        source_expert_order,
        descriptors,
    })
}

fn state_slot(layer: usize, mixer: Mixer, slot: usize) -> StateSlot {
    match mixer {
        Mixer::DeltaNet => StateSlot {
            layer,
            slot,
            domain: StateDomain::DeltaNetConvAndRecurrent,
            device_buffers_required_before_execution: vec![
                "deltanet_conv_history",
                "deltanet_recurrent_state",
            ],
            rollback_buffers_required_before_execution: vec![
                "deltanet_conv_history_rollback",
                "deltanet_recurrent_state_rollback",
            ],
            state_materialized_by_this_plan: false,
        },
        Mixer::Gqa => StateSlot {
            layer,
            slot,
            domain: StateDomain::GqaKv,
            device_buffers_required_before_execution: vec!["gqa_key_cache", "gqa_value_cache"],
            rollback_buffers_required_before_execution: vec![
                "gqa_key_cache_rollback",
                "gqa_value_cache_rollback",
            ],
            state_materialized_by_this_plan: false,
        },
    }
}

fn layer_command_boundary_order(mixer: Mixer, layer: usize) -> Vec<&'static str> {
    let mut order = vec!["input_hidden", "input_rmsnorm"];
    match mixer {
        Mixer::DeltaNet => order.extend([
            "deltanet_qkvz_ba_projection",
            "deltanet_conv_and_recurrent_state",
            "deltanet_gated_rmsnorm",
            "deltanet_out_projection",
        ]),
        Mixer::Gqa => order.extend([
            "gqa_qkv_projection_and_norm",
            "gqa_rope_and_kv_cache",
            "gqa_attention_and_gate",
            "gqa_out_projection",
        ]),
    }
    order.extend([
        "first_residual",
        "post_attention_rmsnorm",
        "router_gate",
        "source_top10_route",
        "all_ten_routed_expert_gate_up_down",
        "routed_expert_weighted_combine",
        "shared_expert_gate_up_down",
        "shared_expert_scalar_gate",
        "moe_combine",
        "second_residual",
    ]);
    if layer == 0 {
        order.insert(0, "l0_embedding_hidden_boundary");
    }
    order
}

fn graph_state_boundaries() -> Vec<GraphStateBoundary> {
    let boundaries = [
        (
            "embedding_to_l0_hidden",
            "embedding_row_gather",
            "layer0_input_rmsnorm",
            vec![HIDDEN],
        ),
        (
            "l0_input_norm_to_deltanet",
            "layer0_input_rmsnorm",
            "layer0_deltanet",
            vec![HIDDEN],
        ),
        (
            "l0_deltanet_to_first_residual",
            "layer0_deltanet_out_projection",
            "layer0_first_residual",
            vec![HIDDEN],
        ),
        (
            "l0_first_residual_to_postnorm",
            "layer0_first_residual",
            "layer0_post_attention_rmsnorm",
            vec![HIDDEN],
        ),
        (
            "l0_postnorm_to_router",
            "layer0_post_attention_rmsnorm",
            "layer0_router_gate",
            vec![HIDDEN],
        ),
        (
            "l0_router_to_top10_route",
            "layer0_router_gate",
            "layer0_source_top10_route",
            vec![EXPERTS],
        ),
        (
            "l0_route_to_routed_moe",
            "layer0_source_top10_route",
            "layer0_all_ten_routed_expert_wave",
            vec![TOP_K],
        ),
        (
            "l0_routed_moe_to_combine",
            "layer0_all_ten_routed_expert_wave",
            "layer0_moe_combine",
            vec![TOP_K, HIDDEN],
        ),
        (
            "l0_shared_expert_to_combine",
            "layer0_shared_expert",
            "layer0_moe_combine",
            vec![HIDDEN],
        ),
        (
            "l0_moe_combine_to_second_residual",
            "layer0_moe_combine",
            "layer0_second_residual",
            vec![HIDDEN],
        ),
        (
            "layer47_second_residual_to_final_norm",
            "layer47_second_residual",
            "final_rmsnorm",
            vec![HIDDEN],
        ),
        (
            "final_norm_to_all_row_lm_head",
            "final_rmsnorm",
            "all_row_lm_head",
            vec![HIDDEN],
        ),
        (
            "all_row_lm_head_to_tail_mask",
            "all_row_lm_head",
            "reserved_tail_mask",
            vec![VOCAB],
        ),
        (
            "tail_mask_to_deterministic_sample",
            "reserved_tail_mask",
            "deterministic_sample",
            vec![TOKENIZER_VOCAB],
        ),
        (
            "sample_to_tokenizer_feedback",
            "deterministic_sample",
            "tokenizer_feedback",
            vec![1],
        ),
    ];
    boundaries
        .into_iter()
        .enumerate()
        .map(
            |(ordinal, (name, producer, consumer, shape))| GraphStateBoundary {
                ordinal,
                name,
                producer,
                consumer,
                shape,
                required_for_future_graph: true,
                materialized_by_this_plan: false,
            },
        )
        .collect()
}

fn full_command_graph_order() -> Vec<String> {
    let mut order = Vec::with_capacity(LAYERS + 6);
    order.push("embedding".into());
    order.extend((0..LAYERS).map(|layer| format!("layer_{layer}")));
    order.extend([
        "final_rmsnorm".into(),
        "all_row_lm_head".into(),
        "reserved_tail_mask".into(),
        "deterministic_sample".into(),
        "tokenizer_feedback".into(),
    ]);
    order
}

fn claim_boundary() -> ClaimBoundary {
    ClaimBoundary {
        assembly_authority_only: true,
        decoder_readiness_report: false,
        artifact_payload_open_or_scan_performed: false,
        metal_device_or_dispatch_performed: false,
        runtime_watcher_registry_server_or_hcli_changed: false,
        model_execution_performed: false,
        token_generation_or_feedback_performed: false,
        tps_or_tg_measured: false,
        execution_status: EXECUTION_STATUS,
    }
}

fn build_authority(
    inventory_document: SealedDocument,
    config_document: SealedDocument,
) -> Result<PayloadScheduleAuthority, String> {
    let inventory: ManifestInventory = serde_json::from_value(inventory_document.value.clone())
        .map_err(|error| format!("cannot decode descriptor inventory authority: {error}"))?;
    let config: SourceConfigAuthority = serde_json::from_value(config_document.value.clone())
        .map_err(|error| format!("cannot decode source-config authority: {error}"))?;
    validate_inventory(&inventory)?;
    expect_exact_config(&config)?;
    if inventory.source.repository != config.source.repository {
        return Err(
            "descriptor inventory and source-config authority repository bindings disagree".into(),
        );
    }

    let mut index = BTreeMap::new();
    for (ordinal, descriptor) in inventory.tensors.iter().enumerate() {
        if index
            .insert(descriptor.tensor_name.as_str(), (ordinal, descriptor))
            .is_some()
        {
            return Err(format!(
                "descriptor inventory contains duplicate tensor name {}",
                descriptor.tensor_name
            ));
        }
    }
    if index.len() != inventory.tensors.len() {
        return Err("descriptor inventory name index is incomplete".into());
    }

    let mut used = BTreeSet::new();
    let embedding = bind_tensor(
        &index,
        &mut used,
        "embedding",
        "model.embed_tokens.weight".into(),
        &[VOCAB, HIDDEN],
    )?;
    let mut layers = Vec::with_capacity(LAYERS);
    let mut deltanet_slots = Vec::with_capacity(DELTANET_LAYERS);
    let mut gqa_slots = Vec::with_capacity(GQA_LAYERS);

    for layer in 0..LAYERS {
        let mixer = Mixer::expected(layer);
        let slot = match mixer {
            Mixer::DeltaNet => deltanet_slots.len(),
            Mixer::Gqa => gqa_slots.len(),
        };
        let state_slot = state_slot(layer, mixer, slot);
        match mixer {
            Mixer::DeltaNet => deltanet_slots.push(state_slot.clone()),
            Mixer::Gqa => gqa_slots.push(state_slot.clone()),
        }
        let prefix = format!("model.layers.{layer}");
        let input_layernorm = bind_tensor(
            &index,
            &mut used,
            "input_layernorm",
            format!("{prefix}.input_layernorm.weight"),
            &[HIDDEN],
        )?;
        let specifications = match mixer {
            Mixer::DeltaNet => deltanet_tensor_specifications(layer),
            Mixer::Gqa => gqa_tensor_specifications(layer),
        };
        let mixer_tensor_execution_order = specifications
            .iter()
            .map(|(role, _, _)| *role)
            .collect::<Vec<_>>();
        let mixer_tensors = specifications
            .into_iter()
            .map(|(role, name, shape)| bind_tensor(&index, &mut used, role, name, &shape))
            .collect::<Result<Vec<_>, _>>()?;
        let post_attention_layernorm = bind_tensor(
            &index,
            &mut used,
            "post_attention_layernorm",
            format!("{prefix}.post_attention_layernorm.weight"),
            &[HIDDEN],
        )?;
        let router_gate = bind_tensor(
            &index,
            &mut used,
            "router_gate",
            format!("{prefix}.mlp.gate.weight"),
            &[EXPERTS, HIDDEN],
        )?;
        let routed_experts = RoutedExpertPayloads {
            expert_count: EXPERTS,
            top_k: TOP_K,
            projection_execution_order: ["gate_proj", "up_proj", "down_proj"],
            tables: vec![
                expert_projection_table(
                    layer,
                    "gate_proj",
                    vec![INTERMEDIATE, HIDDEN],
                    &index,
                    &mut used,
                )?,
                expert_projection_table(
                    layer,
                    "up_proj",
                    vec![INTERMEDIATE, HIDDEN],
                    &index,
                    &mut used,
                )?,
                expert_projection_table(
                    layer,
                    "down_proj",
                    vec![HIDDEN, INTERMEDIATE],
                    &index,
                    &mut used,
                )?,
            ],
            route_selection_materialized_by_this_plan: false,
            expert_payload_opened_by_this_plan: false,
        };
        let shared_expert = SharedExpertPayloads {
            execution_order: ["gate_proj", "up_proj", "down_proj", "scalar_gate"],
            gate_proj: bind_tensor(
                &index,
                &mut used,
                "shared_expert_gate_proj",
                format!("{prefix}.mlp.shared_expert.gate_proj.weight"),
                &[INTERMEDIATE, HIDDEN],
            )?,
            up_proj: bind_tensor(
                &index,
                &mut used,
                "shared_expert_up_proj",
                format!("{prefix}.mlp.shared_expert.up_proj.weight"),
                &[INTERMEDIATE, HIDDEN],
            )?,
            down_proj: bind_tensor(
                &index,
                &mut used,
                "shared_expert_down_proj",
                format!("{prefix}.mlp.shared_expert.down_proj.weight"),
                &[HIDDEN, INTERMEDIATE],
            )?,
            scalar_gate: bind_tensor(
                &index,
                &mut used,
                "shared_expert_scalar_gate",
                format!("{prefix}.mlp.shared_expert_gate.weight"),
                &[1, HIDDEN],
            )?,
        };
        layers.push(LayerPayloadPlan {
            layer,
            mixer,
            state_slot,
            input_layernorm,
            mixer_tensor_execution_order,
            mixer_tensors,
            post_attention_layernorm,
            router_gate,
            routed_experts,
            shared_expert,
            layer_command_boundary_order: layer_command_boundary_order(mixer, layer),
        });
    }

    let terminal_head = TerminalHeadPlan {
        final_norm: bind_tensor(
            &index,
            &mut used,
            "final_norm",
            "model.norm.weight".into(),
            &[HIDDEN],
        )?,
        lm_head: bind_tensor(
            &index,
            &mut used,
            "all_row_lm_head",
            "lm_head.weight".into(),
            &[VOCAB, HIDDEN],
        )?,
        all_row_lm_head_rows: VOCAB,
        tokenizer_addressable_rows: TOKENIZER_VOCAB,
        reserved_tail_rows: TAIL_ROWS,
        execution_order: [
            "final_rmsnorm",
            "all_row_lm_head",
            "reserved_tail_mask",
            "deterministic_sample",
            "tokenizer_feedback",
        ],
    };
    if used.len() != COMPLETE_TENSOR_COUNT || used.len() != inventory.tensors.len() {
        return Err(format!(
            "scheduled {} descriptors, but the exact Qwen80 inventory has {}",
            used.len(),
            inventory.tensors.len()
        ));
    }
    if deltanet_slots.len() != DELTANET_LAYERS || gqa_slots.len() != GQA_LAYERS {
        return Err("3x DeltaNet / 1x GQA state-slot accounting drifted".into());
    }
    let plan = PayloadScheduleAuthority {
        schema: SCHEMA,
        status: STATUS,
        source_authority: SourceAuthorityBinding {
            model_id: MODEL_ID,
            model_key: MODEL_KEY,
            source_repository: SOURCE_REPOSITORY,
            source_revision: SOURCE_REVISION,
            source_config_sha256: SOURCE_CONFIG_SHA256,
            descriptor_inventory_canonical_path: inventory_document
                .canonical_path
                .display()
                .to_string(),
            descriptor_inventory_document_sha256: inventory_document.document_sha256,
            descriptor_inventory_schema: inventory.schema,
            descriptor_inventory_status: inventory.status,
            descriptor_inventory_seal_sha256: inventory.seal_sha256,
            descriptor_inventory_tensor_count: inventory.tensors.len(),
            source_config_authority_canonical_path: config_document
                .canonical_path
                .display()
                .to_string(),
            source_config_authority_document_sha256: config_document.document_sha256,
            source_config_authority_schema: config.schema,
            source_config_authority_status: config.status,
            source_config_authority_seal_sha256: config.seal_sha256,
        },
        geometry: geometry(),
        embedding,
        layers,
        deltanet_state_slots: deltanet_slots,
        gqa_state_slots: gqa_slots,
        terminal_head,
        future_graph_state_boundaries: graph_state_boundaries(),
        full_command_graph_order: full_command_graph_order(),
        resolved_tensor_binding_count: used.len(),
        all_48_layers_scheduled: true,
        all_descriptors_source_artifact_bound: true,
        claim_boundary: claim_boundary(),
    };
    validate_plan(&plan)?;
    Ok(plan)
}

fn expected_mixer_order(mixer: Mixer) -> Vec<&'static str> {
    match mixer {
        Mixer::DeltaNet => vec![
            "deltanet_in_proj_qkvz",
            "deltanet_in_proj_ba",
            "deltanet_causal_conv1d",
            "deltanet_a_log",
            "deltanet_dt_bias",
            "deltanet_gated_rmsnorm",
            "deltanet_out_proj",
        ],
        Mixer::Gqa => vec![
            "gqa_q_proj",
            "gqa_k_proj",
            "gqa_v_proj",
            "gqa_q_norm",
            "gqa_k_norm",
            "gqa_o_proj",
        ],
    }
}

fn validate_binding(
    binding: &TensorBinding,
    expected_name: &str,
    expected_shape: &[usize],
    label: &str,
) -> Result<(), String> {
    if binding.tensor_name != expected_name
        || binding.shape != expected_shape
        || binding.elements != checked_elements(expected_shape, label)?
        || binding.artifact_path.is_empty()
        || binding.artifact_bytes == 0
        || binding.source_dtype != "BF16"
        || !is_lower_sha256(&binding.artifact_sha256)
        || !is_lower_sha256(&binding.source_shard_sha256)
        || binding.layout
            != (PackedLayout {
                magic: "HQ30G1B1".into(),
                group_size: GROUP_SIZE,
                scale_dtype: "float16".into(),
                sign_bit_order: "little".into(),
                version: 1,
            })
    {
        return Err(format!("{label} binding identity or geometry drifted"));
    }
    Ok(())
}

fn validate_layer(layer: &LayerPayloadPlan) -> Result<(), String> {
    if layer.layer >= LAYERS || layer.mixer != Mixer::expected(layer.layer) {
        return Err(format!("layer {} hybrid schedule drifted", layer.layer));
    }
    let expected_slot = layer.layer / 4 * 3 + layer.layer % 4;
    let (expected_domain, expected_state_slot) = match layer.mixer {
        Mixer::DeltaNet => (StateDomain::DeltaNetConvAndRecurrent, expected_slot),
        Mixer::Gqa => (StateDomain::GqaKv, layer.layer / 4),
    };
    if layer.state_slot.layer != layer.layer
        || layer.state_slot.slot != expected_state_slot
        || layer.state_slot.domain != expected_domain
        || layer.state_slot.state_materialized_by_this_plan
    {
        return Err(format!("layer {} state-slot mapping drifted", layer.layer));
    }
    validate_binding(
        &layer.input_layernorm,
        &format!("model.layers.{}.input_layernorm.weight", layer.layer),
        &[HIDDEN],
        "input layernorm",
    )?;
    let expected_specs = match layer.mixer {
        Mixer::DeltaNet => deltanet_tensor_specifications(layer.layer),
        Mixer::Gqa => gqa_tensor_specifications(layer.layer),
    };
    if layer.mixer_tensor_execution_order != expected_mixer_order(layer.mixer)
        || layer.mixer_tensors.len() != expected_specs.len()
    {
        return Err(format!("layer {} mixer tensor order drifted", layer.layer));
    }
    for (binding, (role, name, shape)) in layer.mixer_tensors.iter().zip(expected_specs) {
        if binding.role != role {
            return Err(format!("layer {} mixer role order drifted", layer.layer));
        }
        validate_binding(binding, &name, &shape, "mixer")?;
    }
    let prefix = format!("model.layers.{}", layer.layer);
    validate_binding(
        &layer.post_attention_layernorm,
        &format!("{prefix}.post_attention_layernorm.weight"),
        &[HIDDEN],
        "post-attention layernorm",
    )?;
    validate_binding(
        &layer.router_gate,
        &format!("{prefix}.mlp.gate.weight"),
        &[EXPERTS, HIDDEN],
        "router",
    )?;
    if layer.routed_experts.expert_count != EXPERTS
        || layer.routed_experts.top_k != TOP_K
        || layer.routed_experts.projection_execution_order != ["gate_proj", "up_proj", "down_proj"]
        || layer
            .routed_experts
            .route_selection_materialized_by_this_plan
        || layer.routed_experts.expert_payload_opened_by_this_plan
        || layer.routed_experts.tables.len() != 3
    {
        return Err(format!(
            "layer {} routed-expert authority drifted",
            layer.layer
        ));
    }
    for (table, (projection, shape)) in layer.routed_experts.tables.iter().zip([
        ("gate_proj", vec![INTERMEDIATE, HIDDEN]),
        ("up_proj", vec![INTERMEDIATE, HIDDEN]),
        ("down_proj", vec![HIDDEN, INTERMEDIATE]),
    ]) {
        if table.projection != projection
            || table.expected_shape != shape
            || table.descriptors.len() != EXPERTS
            || table.source_expert_order != (0..EXPERTS).collect::<Vec<_>>()
        {
            return Err(format!(
                "layer {} {projection} expert order drifted",
                layer.layer
            ));
        }
        for (expert, binding) in table.descriptors.iter().enumerate() {
            if binding.role != format!("routed_expert_{projection}") {
                return Err(format!(
                    "layer {} {projection} expert role drifted",
                    layer.layer
                ));
            }
            validate_binding(
                binding,
                &format!("{prefix}.mlp.experts.{expert}.{projection}.weight"),
                &shape,
                "routed expert",
            )?;
        }
    }
    if layer.shared_expert.execution_order != ["gate_proj", "up_proj", "down_proj", "scalar_gate"] {
        return Err(format!("layer {} shared-expert order drifted", layer.layer));
    }
    validate_binding(
        &layer.shared_expert.gate_proj,
        &format!("{prefix}.mlp.shared_expert.gate_proj.weight"),
        &[INTERMEDIATE, HIDDEN],
        "shared gate",
    )?;
    validate_binding(
        &layer.shared_expert.up_proj,
        &format!("{prefix}.mlp.shared_expert.up_proj.weight"),
        &[INTERMEDIATE, HIDDEN],
        "shared up",
    )?;
    validate_binding(
        &layer.shared_expert.down_proj,
        &format!("{prefix}.mlp.shared_expert.down_proj.weight"),
        &[HIDDEN, INTERMEDIATE],
        "shared down",
    )?;
    validate_binding(
        &layer.shared_expert.scalar_gate,
        &format!("{prefix}.mlp.shared_expert_gate.weight"),
        &[1, HIDDEN],
        "shared scalar gate",
    )?;
    if layer.layer_command_boundary_order != layer_command_boundary_order(layer.mixer, layer.layer)
    {
        return Err(format!(
            "layer {} command-state boundary order drifted",
            layer.layer
        ));
    }
    Ok(())
}

fn validate_plan(plan: &PayloadScheduleAuthority) -> Result<(), String> {
    if plan.schema != SCHEMA || plan.status != STATUS || plan.geometry != geometry() {
        return Err("payload/schedule authority schema, status, or geometry drifted".into());
    }
    let source = &plan.source_authority;
    if source.model_id != MODEL_ID
        || source.model_key != MODEL_KEY
        || source.source_repository != SOURCE_REPOSITORY
        || source.source_revision != SOURCE_REVISION
        || source.source_config_sha256 != SOURCE_CONFIG_SHA256
        || source.descriptor_inventory_schema != MANIFEST_SCHEMA
        || source.descriptor_inventory_status != MANIFEST_STATUS
        || source.source_config_authority_schema != SOURCE_CONFIG_SCHEMA
        || source.source_config_authority_status != SOURCE_CONFIG_STATUS
        || source.descriptor_inventory_tensor_count != COMPLETE_TENSOR_COUNT
        || !is_lower_sha256(&source.descriptor_inventory_document_sha256)
        || !is_lower_sha256(&source.descriptor_inventory_seal_sha256)
        || !is_lower_sha256(&source.source_config_authority_document_sha256)
        || !is_lower_sha256(&source.source_config_authority_seal_sha256)
    {
        return Err("source/config/descriptor authority binding drifted".into());
    }
    validate_binding(
        &plan.embedding,
        "model.embed_tokens.weight",
        &[VOCAB, HIDDEN],
        "embedding",
    )?;
    if plan.layers.len() != LAYERS
        || plan.deltanet_state_slots.len() != DELTANET_LAYERS
        || plan.gqa_state_slots.len() != GQA_LAYERS
        || !plan.all_48_layers_scheduled
        || !plan.all_descriptors_source_artifact_bound
        || plan.resolved_tensor_binding_count != expected_tensor_count()
    {
        return Err("48-layer schedule or resolved descriptor count drifted".into());
    }
    for (expected_layer, layer) in plan.layers.iter().enumerate() {
        if layer.layer != expected_layer {
            return Err("layer array is not source layer order 0 through 47".into());
        }
        validate_layer(layer)?;
    }
    for (slot, state) in plan.deltanet_state_slots.iter().enumerate() {
        if state.domain != StateDomain::DeltaNetConvAndRecurrent
            || state.slot != slot
            || state.layer != slot / 3 * 4 + slot % 3
        {
            return Err("DeltaNet state-slot sequence drifted".into());
        }
    }
    for (slot, state) in plan.gqa_state_slots.iter().enumerate() {
        if state.domain != StateDomain::GqaKv || state.slot != slot || state.layer != slot * 4 + 3 {
            return Err("GQA state-slot sequence drifted".into());
        }
    }
    let terminal = &plan.terminal_head;
    validate_binding(
        &terminal.final_norm,
        "model.norm.weight",
        &[HIDDEN],
        "final norm",
    )?;
    validate_binding(
        &terminal.lm_head,
        "lm_head.weight",
        &[VOCAB, HIDDEN],
        "lm head",
    )?;
    if terminal.all_row_lm_head_rows != VOCAB
        || terminal.tokenizer_addressable_rows != TOKENIZER_VOCAB
        || terminal.reserved_tail_rows != TAIL_ROWS
        || terminal.execution_order
            != [
                "final_rmsnorm",
                "all_row_lm_head",
                "reserved_tail_mask",
                "deterministic_sample",
                "tokenizer_feedback",
            ]
    {
        return Err("terminal-head/tail-mask order drifted".into());
    }
    if plan.future_graph_state_boundaries != graph_state_boundaries()
        || plan.full_command_graph_order != full_command_graph_order()
        || plan.claim_boundary != claim_boundary()
    {
        return Err("future graph boundaries or claim boundary drifted".into());
    }
    Ok(())
}

fn raw_plan_bytes(plan: &PayloadScheduleAuthority) -> Result<Vec<u8>, String> {
    let mut bytes = serde_json::to_vec_pretty(plan)
        .map_err(|error| format!("cannot serialize payload/schedule authority: {error}"))?;
    bytes.push(b'\n');
    Ok(bytes)
}

/// The static plan deliberately has no seal of its own.  A future device lease
/// must pin the raw document SHA after this create-new write; allowing a plan
/// replacement at this path would invalidate that future lease.
fn write_new_raw_plan(path: &Path, plan: &PayloadScheduleAuthority) -> Result<(), String> {
    if !path.is_absolute() {
        return Err("raw plan output path must be absolute".into());
    }
    let bytes = raw_plan_bytes(plan)?;
    let mut output = OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(path)
        .map_err(|error| format!("cannot create new raw plan {}: {error}", path.display()))?;
    output
        .write_all(&bytes)
        .map_err(|error| format!("cannot write raw plan {}: {error}", path.display()))?;
    output
        .sync_all()
        .map_err(|error| format!("cannot sync raw plan {}: {error}", path.display()))?;
    Ok(())
}

fn main() {
    let arguments = env::args().skip(1).collect::<Vec<_>>();
    let args = parse_args(&arguments).unwrap_or_else(|error| {
        eprintln!("{error}");
        std::process::exit(2);
    });
    let inventory = read_sealed_document(&args.descriptor_inventory, "descriptor inventory")
        .unwrap_or_else(|error| panic!("cannot read descriptor inventory: {error}"));
    let config = read_sealed_document(&args.source_config_authority, "source-config authority")
        .unwrap_or_else(|error| panic!("cannot read source-config authority: {error}"));
    let plan = build_authority(inventory, config)
        .unwrap_or_else(|error| panic!("cannot build Qwen80 payload/schedule authority: {error}"));
    write_new_raw_plan(&args.out, &plan)
        .unwrap_or_else(|error| panic!("cannot emit Qwen80 payload/schedule authority: {error}"));
    println!("{}", args.out.display());
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    fn test_sha256(tag: &str) -> String {
        sha256_hex(tag.as_bytes())
    }

    fn layout() -> PackedLayout {
        PackedLayout {
            magic: "HQ30G1B1".into(),
            group_size: GROUP_SIZE,
            scale_dtype: "float16".into(),
            sign_bit_order: "little".into(),
            version: 1,
        }
    }

    fn descriptor(name: String, shape: Vec<usize>, ordinal: usize) -> TensorDescriptor {
        TensorDescriptor {
            tensor_name: name,
            elements: checked_elements(&shape, "test descriptor").unwrap(),
            shape,
            artifact_path: format!("/synthetic/qwen80/{ordinal:05}.hq30g"),
            artifact_bytes: 64 + ordinal as u64,
            artifact_sha256: test_sha256(&format!("artifact-{ordinal}")),
            source_dtype: "BF16".into(),
            source_shard: format!("model-{:05}-of-00040.safetensors", ordinal % 40 + 1),
            source_shard_sha256: test_sha256(&format!("source-shard-{}", ordinal % 40)),
            layout: layout(),
        }
    }

    fn fixture_inventory() -> ManifestInventory {
        let mut tensors = Vec::with_capacity(COMPLETE_TENSOR_COUNT);
        let mut ordinal = 0usize;
        let mut add = |name: String, shape: Vec<usize>| {
            tensors.push(descriptor(name, shape, ordinal));
            ordinal += 1;
        };
        add("model.embed_tokens.weight".into(), vec![VOCAB, HIDDEN]);
        for layer in 0..LAYERS {
            let prefix = format!("model.layers.{layer}");
            add(format!("{prefix}.input_layernorm.weight"), vec![HIDDEN]);
            for (_, name, shape) in match Mixer::expected(layer) {
                Mixer::DeltaNet => deltanet_tensor_specifications(layer),
                Mixer::Gqa => gqa_tensor_specifications(layer),
            } {
                add(name, shape);
            }
            add(
                format!("{prefix}.post_attention_layernorm.weight"),
                vec![HIDDEN],
            );
            add(format!("{prefix}.mlp.gate.weight"), vec![EXPERTS, HIDDEN]);
            for expert in 0..EXPERTS {
                add(
                    format!("{prefix}.mlp.experts.{expert}.gate_proj.weight"),
                    vec![INTERMEDIATE, HIDDEN],
                );
                add(
                    format!("{prefix}.mlp.experts.{expert}.up_proj.weight"),
                    vec![INTERMEDIATE, HIDDEN],
                );
                add(
                    format!("{prefix}.mlp.experts.{expert}.down_proj.weight"),
                    vec![HIDDEN, INTERMEDIATE],
                );
            }
            add(
                format!("{prefix}.mlp.shared_expert.gate_proj.weight"),
                vec![INTERMEDIATE, HIDDEN],
            );
            add(
                format!("{prefix}.mlp.shared_expert.up_proj.weight"),
                vec![INTERMEDIATE, HIDDEN],
            );
            add(
                format!("{prefix}.mlp.shared_expert.down_proj.weight"),
                vec![HIDDEN, INTERMEDIATE],
            );
            add(
                format!("{prefix}.mlp.shared_expert_gate.weight"),
                vec![1, HIDDEN],
            );
        }
        add("model.norm.weight".into(), vec![HIDDEN]);
        add("lm_head.weight".into(), vec![VOCAB, HIDDEN]);
        assert_eq!(tensors.len(), COMPLETE_TENSOR_COUNT);
        ManifestInventory {
            schema: MANIFEST_SCHEMA.into(),
            status: MANIFEST_STATUS.into(),
            seal_sha256: test_sha256("manifest-seal"),
            source: ManifestSource {
                repository: SOURCE_REPOSITORY.into(),
                tensor_count: tensors.len(),
            },
            tensors,
        }
    }

    fn fixture_config() -> SourceConfigAuthority {
        SourceConfigAuthority {
            schema: SOURCE_CONFIG_SCHEMA.into(),
            status: SOURCE_CONFIG_STATUS.into(),
            seal_sha256: test_sha256("config-seal"),
            source: SourceConfigSource {
                repository: SOURCE_REPOSITORY.into(),
                revision: SOURCE_REVISION.into(),
            },
            architecture: SourceArchitecture {
                architectures: vec!["Qwen3NextForCausalLM".into()],
                config_captured: true,
                config_sha256: SOURCE_CONFIG_SHA256.into(),
                hidden_size: HIDDEN,
                model_type: "qwen3_next".into(),
                num_experts: EXPERTS,
                num_experts_per_tok: TOP_K,
                num_hidden_layers: LAYERS,
                vocab_size: VOCAB,
            },
        }
    }

    fn documents() -> (SealedDocument, SealedDocument) {
        let inventory = fixture_inventory();
        let config = fixture_config();
        let inventory_value = serde_json::to_value(&json!({
            "schema": inventory.schema,
            "status": inventory.status,
            "seal_sha256": inventory.seal_sha256,
            "source": {"repository": inventory.source.repository, "tensor_count": inventory.source.tensor_count},
            "tensors": inventory.tensors.iter().map(|descriptor| json!({
                "tensor_name": descriptor.tensor_name,
                "shape": descriptor.shape,
                "elements": descriptor.elements,
                "artifact_path": descriptor.artifact_path,
                "artifact_bytes": descriptor.artifact_bytes,
                "artifact_sha256": descriptor.artifact_sha256,
                "source_dtype": descriptor.source_dtype,
                "source_shard": descriptor.source_shard,
                "source_shard_sha256": descriptor.source_shard_sha256,
                "layout": {"magic": descriptor.layout.magic, "group_size": descriptor.layout.group_size, "scale_dtype": descriptor.layout.scale_dtype, "sign_bit_order": descriptor.layout.sign_bit_order, "version": descriptor.layout.version}
            })).collect::<Vec<_>>()
        }))
        .unwrap();
        let config_value = json!({
            "schema": config.schema,
            "status": config.status,
            "seal_sha256": config.seal_sha256,
            "source": {"repository": config.source.repository, "revision": config.source.revision},
            "architecture": {
                "architectures": config.architecture.architectures,
                "config_captured": config.architecture.config_captured,
                "config_sha256": config.architecture.config_sha256,
                "hidden_size": config.architecture.hidden_size,
                "model_type": config.architecture.model_type,
                "num_experts": config.architecture.num_experts,
                "num_experts_per_tok": config.architecture.num_experts_per_tok,
                "num_hidden_layers": config.architecture.num_hidden_layers,
                "vocab_size": config.architecture.vocab_size
            }
        });
        (
            SealedDocument {
                canonical_path: PathBuf::from("/synthetic/inventory.json"),
                document_sha256: test_sha256("inventory-document"),
                value: inventory_value,
            },
            SealedDocument {
                canonical_path: PathBuf::from("/synthetic/config.json"),
                document_sha256: test_sha256("config-document"),
                value: config_value,
            },
        )
    }

    fn plan() -> PayloadScheduleAuthority {
        let (inventory, config) = documents();
        build_authority(inventory, config).unwrap()
    }

    #[test]
    fn complete_authority_maps_all_48_layers_and_every_tensor() {
        let plan = plan();
        validate_plan(&plan).unwrap();
        assert_eq!(expected_tensor_count(), COMPLETE_TENSOR_COUNT);
        assert_eq!(plan.resolved_tensor_binding_count, COMPLETE_TENSOR_COUNT);
        assert_eq!(plan.layers.len(), 48);
        assert_eq!(plan.deltanet_state_slots.len(), 36);
        assert_eq!(plan.gqa_state_slots.len(), 12);
        assert_eq!(plan.layers[0].mixer, Mixer::DeltaNet);
        assert_eq!(plan.layers[3].mixer, Mixer::Gqa);
        assert_eq!(plan.layers[47].mixer, Mixer::Gqa);
        assert_eq!(
            plan.layers[0].routed_experts.tables[0].descriptors.len(),
            512
        );
        assert_eq!(
            plan.layers[47].routed_experts.tables[2].descriptors.len(),
            512
        );
    }

    #[test]
    fn exact_state_slot_schedule_is_36_deltanet_and_12_gqa() {
        let plan = plan();
        assert_eq!(
            plan.deltanet_state_slots
                .iter()
                .map(|slot| slot.layer)
                .collect::<Vec<_>>(),
            vec![
                0, 1, 2, 4, 5, 6, 8, 9, 10, 12, 13, 14, 16, 17, 18, 20, 21, 22, 24, 25, 26, 28, 29,
                30, 32, 33, 34, 36, 37, 38, 40, 41, 42, 44, 45, 46
            ]
        );
        assert_eq!(
            plan.gqa_state_slots
                .iter()
                .map(|slot| slot.layer)
                .collect::<Vec<_>>(),
            vec![3, 7, 11, 15, 19, 23, 27, 31, 35, 39, 43, 47]
        );
    }

    #[test]
    fn validator_rejects_schedule_or_terminal_order_drift() {
        let mut schedule_plan = plan();
        schedule_plan.layers[3].mixer = Mixer::DeltaNet;
        let error = validate_plan(&schedule_plan).unwrap_err();
        assert!(error.contains("schedule"));

        let mut terminal_plan = plan();
        terminal_plan.terminal_head.execution_order.swap(0, 1);
        let error = validate_plan(&terminal_plan).unwrap_err();
        assert!(error.contains("terminal-head"));
    }

    #[test]
    fn builder_rejects_missing_or_shape_mismatched_descriptor() {
        let (mut inventory, config) = documents();
        let tensors = inventory
            .value
            .get_mut("tensors")
            .and_then(Value::as_array_mut)
            .unwrap();
        tensors
            .iter_mut()
            .find(|tensor| tensor["tensor_name"] == "model.layers.3.self_attn.q_proj.weight")
            .unwrap()["shape"] = json!([1, HIDDEN]);
        let error = build_authority(inventory, config).unwrap_err();
        assert!(error.contains("q_proj") && error.contains("shape"));

        let (mut inventory, config) = documents();
        let tensors = inventory
            .value
            .get_mut("tensors")
            .and_then(Value::as_array_mut)
            .unwrap();
        tensors.retain(|tensor| {
            tensor["tensor_name"] != "model.layers.0.mlp.shared_expert_gate.weight"
        });
        let error = build_authority(inventory, config).unwrap_err();
        assert!(error.contains("tensor count"));
    }

    #[test]
    fn claim_boundary_is_prepared_not_executed_and_not_readiness() {
        let plan = plan();
        assert_eq!(plan.status, STATUS);
        assert!(plan.claim_boundary.assembly_authority_only);
        assert!(!plan.claim_boundary.decoder_readiness_report);
        assert!(!plan.claim_boundary.artifact_payload_open_or_scan_performed);
        assert!(!plan.claim_boundary.metal_device_or_dispatch_performed);
        assert!(!plan.claim_boundary.model_execution_performed);
        assert!(!plan.claim_boundary.token_generation_or_feedback_performed);
        assert!(!plan.claim_boundary.tps_or_tg_measured);
        assert_eq!(plan.claim_boundary.execution_status, EXECUTION_STATUS);
    }

    #[test]
    fn l0_and_terminal_boundaries_are_explicit_and_ordered() {
        let plan = plan();
        let names = plan
            .future_graph_state_boundaries
            .iter()
            .map(|boundary| boundary.name)
            .collect::<Vec<_>>();
        assert_eq!(names[0], "embedding_to_l0_hidden");
        assert!(names.contains(&"l0_first_residual_to_postnorm"));
        assert!(names.contains(&"l0_postnorm_to_router"));
        assert!(names.contains(&"l0_route_to_routed_moe"));
        assert!(names.contains(&"l0_shared_expert_to_combine"));
        assert_eq!(names[names.len() - 1], "sample_to_tokenizer_feedback");
        assert_eq!(plan.full_command_graph_order[0], "embedding");
        assert_eq!(plan.full_command_graph_order[48], "layer_47");
        assert_eq!(plan.full_command_graph_order[49], "final_rmsnorm");
    }

    #[test]
    fn raw_plan_output_is_create_new_and_unsealed() {
        let plan = plan();
        let directory = tempfile::tempdir().unwrap();
        let absolute = fs::canonicalize(directory.path())
            .unwrap()
            .join("qwen80-48-layer-authority.json");
        write_new_raw_plan(&absolute, &plan).unwrap();
        let written = fs::read(&absolute).unwrap();
        let value: Value = serde_json::from_slice(&written).unwrap();
        assert_eq!(value["schema"], SCHEMA);
        assert_eq!(value["status"], STATUS);
        assert!(value.get("seal_sha256").is_none());
        let error = write_new_raw_plan(&absolute, &plan).unwrap_err();
        assert!(error.contains("cannot create new raw plan"));
    }

    #[test]
    fn seal_verifier_refuses_tampered_input_documents() {
        let mut value = json!({"schema": "fixture", "counter": 1});
        let seal = sha256_hex(&serde_json::to_vec(&value).unwrap());
        value["seal_sha256"] = Value::String(seal);
        verify_document_seal(&value, "fixture").unwrap();
        value["counter"] = json!(2);
        let error = verify_document_seal(&value, "fixture").unwrap_err();
        assert!(error.contains("does not match"));
    }
}

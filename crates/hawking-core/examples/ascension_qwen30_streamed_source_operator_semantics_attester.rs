#![allow(dead_code)] // The executable exposes small fixture helpers for its CPU-only tests.

//! Metadata-only Qwen30 source-operator and accumulation semantics attester.
//!
//! This program is intentionally *not* a source inference implementation.  It
//! reads a pinned source `config.json`, a pinned local
//! `modeling_qwen3_moe.py`, the sealed metadata-only range authority, and the
//! sealed 369-prefix/forced-token replay contract.  It never accepts a source
//! root or a safetensors path, never opens source tensor payload bytes, never
//! instantiates a source model, and has no GPU, server, HCLI, or lease path.
//!
//! Its result is a fail-closed specification of what a later, separately
//! leased source execution must prove.  In particular, source-code text
//! marker checks are not a numerical-equivalence claim: decorators and
//! selectable attention backends must be bound by fresh execution evidence.

use serde_json::{json, Map, Value};
use sha2::{Digest, Sha256};
use std::collections::BTreeSet;
use std::env;
use std::fs::{self, File, OpenOptions};
use std::io::{Read, Write};
use std::path::{Path, PathBuf};
use std::process;

const RESULT_SCHEMA: &str =
    "hawking.ascension.qwen30_layer_streamed_source_operator_accumulation_semantics_attester.v1";
const RESULT_STATUS: &str =
    "PREPARED_QWEN30_LAYER_STREAMED_SOURCE_OPERATOR_ACCUMULATION_SEMANTICS_NOT_EXECUTED";
const FUTURE_ATTESTATION_SCHEMA: &str =
    "hawking.ascension.qwen30_layer_streamed_source_operator_accumulation_execution_attestation.v1";
const FUTURE_ATTESTATION_STATUS: &str =
    "EARNED_QWEN30_LAYER_STREAMED_SOURCE_OPERATOR_ACCUMULATION_SEMANTICS_ATTESTED";
const RANGE_AUTHORITY_SCHEMA: &str =
    "hawking.ascension.qwen30_streamed_oracle_metadata_only_range_map_authority.v1";
const RANGE_AUTHORITY_STATUS: &str =
    "PREPARED_QWEN30_STREAMED_ORACLE_SOURCE_RANGE_MAP_AUTHORITY_NOT_EXECUTED";
const TRACE_CONTRACT_SCHEMA: &str =
    "hawking.ascension.qwen30_hq30gr2_source_bf16_three_way_final_logit_contract.v1";
const TRACE_CONTRACT_STATUS: &str =
    "PREPARED_SOURCE_BF16_THREE_WAY_FINAL_LOGIT_DISTANCE_CONTRACT_NOT_RUN";

const SOURCE_MODEL_ID: &str = "Qwen3-Coder-30B-A3B-Instruct";
const SOURCE_REVISION: &str = "b2cff646eb4bb1d68355c01b18ae02e7cf42d120";
const SOURCE_CONFIG_SHA256: &str =
    "e2c8d8eea39471785cd93379d8b48241ad7dcda299013155dd02526e34a0de62";
const SOURCE_MODELING_SHA256: &str =
    "f07ab1b3a17e27612006a447812b83667af7ab3d527afe3ebbdf89dc5dac8a90";
const RANGE_AUTHORITY_CONTENT_SHA256: &str =
    "a17d7014150d8a54ccaa850fa69a76321109b9816860c0193ef055e7ab394a78";
const TRACE_CONTRACT_SEAL_SHA256: &str =
    "efdacf5952583dc03d2aee37b73a0af284f2d865a3e36b9739b16150efe3f726";
const SOURCE_INDEX_SHA256: &str =
    "8dde190b862c7c80ec7403c6495de00c60bbaf246ed479cee4506284989c584c";
const TEMPLATE_TOKEN_IDS_U32LE_SHA256: &str =
    "2e13acf3ddb7d520ca4120b2e3956eead91651ea492b308a4f0fa28a5be66699";

const PREFIX_TOKEN_COUNT: u64 = 369;
const FORCED_TOKEN_ID: u64 = 949;
const FORWARDS_PER_ARM: u64 = PREFIX_TOKEN_COUNT + 1;
const LAYER_COUNT: u64 = 48;
const HIDDEN_SIZE: u64 = 2_048;
const VOCAB_SIZE: u64 = 151_936;
const ATTENTION_HEADS: u64 = 32;
const KV_HEADS: u64 = 4;
const HEAD_DIM: u64 = 128;
const EXPERT_COUNT: u64 = 128;
const TOP_K: u64 = 8;
const MOE_INTERMEDIATE: u64 = 768;
const SOURCE_TENSOR_COUNT: u64 = 18_867;
const SOURCE_SHARD_COUNT: u64 = 16;
const F32_FINAL_LOGIT_BYTES: u64 = VOCAB_SIZE * 4;

const MAX_CONFIG_BYTES: u64 = 1024 * 1024;
const MAX_MODELING_CODE_BYTES: u64 = 8 * 1024 * 1024;
const MAX_RANGE_AUTHORITY_BYTES: u64 = 32 * 1024 * 1024;
const MAX_TRACE_CONTRACT_BYTES: u64 = 4 * 1024 * 1024;

#[derive(Debug)]
struct Args {
    source_config: PathBuf,
    modeling_code: PathBuf,
    range_authority: PathBuf,
    trace_contract: PathBuf,
    source_revision: String,
    out: PathBuf,
}

#[derive(Clone, Copy)]
struct Pins<'a> {
    source_revision: &'a str,
    source_config_sha256: &'a str,
    source_modeling_sha256: &'a str,
    range_authority_content_sha256: &'a str,
    trace_contract_seal_sha256: &'a str,
    source_index_sha256: &'a str,
    template_token_ids_u32le_sha256: &'a str,
}

const PRODUCTION_PINS: Pins<'static> = Pins {
    source_revision: SOURCE_REVISION,
    source_config_sha256: SOURCE_CONFIG_SHA256,
    source_modeling_sha256: SOURCE_MODELING_SHA256,
    range_authority_content_sha256: RANGE_AUTHORITY_CONTENT_SHA256,
    trace_contract_seal_sha256: TRACE_CONTRACT_SEAL_SHA256,
    source_index_sha256: SOURCE_INDEX_SHA256,
    template_token_ids_u32le_sha256: TEMPLATE_TOKEN_IDS_U32LE_SHA256,
};

fn usage() -> &'static str {
    "usage: ascension_qwen30_streamed_source_operator_semantics_attester --source-config ABSOLUTE_CONFIG_JSON --modeling-code ABSOLUTE_MODELING_QWEN3_MOE_PY --range-authority ABSOLUTE_RANGE_AUTHORITY_JSON --trace-contract ABSOLUTE_SEALED_TRACE_CONTRACT_JSON --source-revision PINNED_40_HEX_REVISION --out NEW_ABSOLUTE_JSON"
}

fn parse_args_from<I>(arguments: I) -> Result<Args, String>
where
    I: IntoIterator<Item = String>,
{
    let mut source_config = None;
    let mut modeling_code = None;
    let mut range_authority = None;
    let mut trace_contract = None;
    let mut source_revision = None;
    let mut out = None;
    let mut values = arguments.into_iter();
    while let Some(flag) = values.next() {
        let destination = match flag.as_str() {
            "--source-config" => &mut source_config,
            "--modeling-code" => &mut modeling_code,
            "--range-authority" => &mut range_authority,
            "--trace-contract" => &mut trace_contract,
            "--source-revision" => &mut source_revision,
            "--out" => &mut out,
            "--help" | "-h" => return Err(usage().to_owned()),
            _ => return Err(format!("unsupported option {flag:?}; {}", usage())),
        };
        let value = values
            .next()
            .ok_or_else(|| format!("missing value for {flag}; {}", usage()))?;
        if destination.replace(value).is_some() {
            return Err(format!("{flag} was supplied more than once; {}", usage()));
        }
    }
    let source_config = PathBuf::from(
        source_config.ok_or_else(|| format!("--source-config is required; {}", usage()))?,
    );
    let modeling_code = PathBuf::from(
        modeling_code.ok_or_else(|| format!("--modeling-code is required; {}", usage()))?,
    );
    let range_authority = PathBuf::from(
        range_authority.ok_or_else(|| format!("--range-authority is required; {}", usage()))?,
    );
    let trace_contract = PathBuf::from(
        trace_contract.ok_or_else(|| format!("--trace-contract is required; {}", usage()))?,
    );
    let source_revision =
        source_revision.ok_or_else(|| format!("--source-revision is required; {}", usage()))?;
    let out = PathBuf::from(out.ok_or_else(|| format!("--out is required; {}", usage()))?);
    for (path, label) in [
        (&source_config, "--source-config"),
        (&modeling_code, "--modeling-code"),
        (&range_authority, "--range-authority"),
        (&trace_contract, "--trace-contract"),
        (&out, "--out"),
    ] {
        if !path.is_absolute() {
            return Err(format!("{label} must be absolute"));
        }
    }
    if !is_pinned_revision(&source_revision) {
        return Err("--source-revision must be exactly 40 lowercase hexadecimal characters".into());
    }
    Ok(Args {
        source_config,
        modeling_code,
        range_authority,
        trace_contract,
        source_revision,
        out,
    })
}

fn parse_args() -> Result<Args, String> {
    parse_args_from(env::args().skip(1))
}

fn is_pinned_revision(value: &str) -> bool {
    value.len() == 40
        && value
            .bytes()
            .all(|byte| byte.is_ascii_hexdigit() && !byte.is_ascii_uppercase())
}

fn sha256_hex(bytes: &[u8]) -> String {
    format!("{:x}", Sha256::digest(bytes))
}

fn is_sha256(value: &str) -> bool {
    value.len() == 64 && value.bytes().all(|byte| byte.is_ascii_hexdigit())
}

/// There is intentionally no generic file reader in this target.  Every
/// readable input is a small metadata/code document with an explicit cap and
/// a basename guard; a safetensors file cannot become an input by accident.
fn read_regular_metadata_file(
    path: &Path,
    label: &str,
    maximum_bytes: u64,
    required_basename: Option<&str>,
) -> Result<Vec<u8>, String> {
    if !path.is_absolute() {
        return Err(format!("{label} must be absolute"));
    }
    if let Some(name) = required_basename {
        if path.file_name().and_then(|item| item.to_str()) != Some(name) {
            return Err(format!("{label} must be named {name}"));
        }
    }
    let metadata = fs::symlink_metadata(path)
        .map_err(|error| format!("cannot stat {label} {}: {error}", path.display()))?;
    if metadata.file_type().is_symlink() || !metadata.file_type().is_file() {
        return Err(format!("{label} must be a regular non-symlink file"));
    }
    let length = metadata.len();
    if length == 0 || length > maximum_bytes {
        return Err(format!(
            "{label} must be 1..={maximum_bytes} metadata/code bytes, observed {length}"
        ));
    }
    let mut bytes =
        vec![0u8; usize::try_from(length).map_err(|_| format!("{label} is too large"))?];
    let mut file = File::open(path)
        .map_err(|error| format!("cannot open {label} {}: {error}", path.display()))?;
    file.read_exact(&mut bytes)
        .map_err(|error| format!("cannot read {label} {}: {error}", path.display()))?;
    let restat = fs::symlink_metadata(path)
        .map_err(|error| format!("cannot restat {label} {}: {error}", path.display()))?;
    if restat.file_type().is_symlink() || !restat.file_type().is_file() || restat.len() != length {
        return Err(format!("{label} changed while it was read"));
    }
    Ok(bytes)
}

fn require_extension(path: &Path, expected: &str, label: &str) -> Result<(), String> {
    if path.extension().and_then(|item| item.to_str()) != Some(expected) {
        return Err(format!(
            "{label} must have .{expected} extension; payload paths are not accepted"
        ));
    }
    Ok(())
}

fn parse_json(bytes: &[u8], label: &str) -> Result<Value, String> {
    serde_json::from_slice(bytes).map_err(|error| format!("{label} is not valid JSON: {error}"))
}

fn object<'a>(value: &'a Value, label: &str) -> Result<&'a Map<String, Value>, String> {
    value
        .as_object()
        .ok_or_else(|| format!("{label} must be an object"))
}

fn array<'a>(value: &'a Value, label: &str) -> Result<&'a [Value], String> {
    value
        .as_array()
        .map(Vec::as_slice)
        .ok_or_else(|| format!("{label} must be an array"))
}

fn required<'a>(map: &'a Map<String, Value>, key: &str, label: &str) -> Result<&'a Value, String> {
    map.get(key)
        .ok_or_else(|| format!("{label}.{key} is required"))
}

fn text<'a>(value: &'a Value, label: &str) -> Result<&'a str, String> {
    value
        .as_str()
        .filter(|value| !value.is_empty())
        .ok_or_else(|| format!("{label} must be a non-empty string"))
}

fn boolean(value: &Value, label: &str) -> Result<bool, String> {
    value
        .as_bool()
        .ok_or_else(|| format!("{label} must be boolean"))
}

fn u64_value(value: &Value, label: &str) -> Result<u64, String> {
    value
        .as_u64()
        .ok_or_else(|| format!("{label} must be an unsigned integer"))
}

fn require_text(
    map: &Map<String, Value>,
    key: &str,
    expected: &str,
    label: &str,
) -> Result<(), String> {
    let actual = text(required(map, key, label)?, &format!("{label}.{key}"))?;
    if actual != expected {
        return Err(format!(
            "{label}.{key} drifted: expected {expected:?}, observed {actual:?}"
        ));
    }
    Ok(())
}

fn require_bool(
    map: &Map<String, Value>,
    key: &str,
    expected: bool,
    label: &str,
) -> Result<(), String> {
    let actual = boolean(required(map, key, label)?, &format!("{label}.{key}"))?;
    if actual != expected {
        return Err(format!(
            "{label}.{key} drifted: expected {expected}, observed {actual}"
        ));
    }
    Ok(())
}

fn require_u64(
    map: &Map<String, Value>,
    key: &str,
    expected: u64,
    label: &str,
) -> Result<(), String> {
    let actual = u64_value(required(map, key, label)?, &format!("{label}.{key}"))?;
    if actual != expected {
        return Err(format!(
            "{label}.{key} drifted: expected {expected}, observed {actual}"
        ));
    }
    Ok(())
}

/// Python's `lab.receipts.seal` hashes JSON with sorted keys and compact
/// separators.  Implement the same narrow canonical form here so that a
/// resealed-but-modified replay contract is rejected before it can bind a
/// future oracle run.
fn canonical_json(value: &Value) -> Result<String, String> {
    match value {
        Value::Null => Ok("null".to_owned()),
        Value::Bool(value) => Ok(value.to_string()),
        Value::Number(value) => Ok(value.to_string()),
        Value::String(value) => serde_json::to_string(value)
            .map_err(|error| format!("cannot canonicalize JSON string: {error}")),
        Value::Array(values) => {
            let mut output = String::from("[");
            for (index, item) in values.iter().enumerate() {
                if index != 0 {
                    output.push(',');
                }
                output.push_str(&canonical_json(item)?);
            }
            output.push(']');
            Ok(output)
        }
        Value::Object(values) => {
            let mut keys = values.keys().collect::<Vec<_>>();
            keys.sort_unstable();
            let mut output = String::from("{");
            for (index, key) in keys.into_iter().enumerate() {
                if index != 0 {
                    output.push(',');
                }
                output.push_str(
                    &serde_json::to_string(key)
                        .map_err(|error| format!("cannot canonicalize JSON key: {error}"))?,
                );
                output.push(':');
                output.push_str(&canonical_json(
                    values
                        .get(key)
                        .ok_or_else(|| "canonical JSON key disappeared".to_owned())?,
                )?);
            }
            output.push('}');
            Ok(output)
        }
    }
}

fn canonical_sha256(value: &Value) -> Result<String, String> {
    Ok(sha256_hex(canonical_json(value)?.as_bytes()))
}

fn verify_seal(document: &Value, label: &str) -> Result<String, String> {
    let root = object(document, label)?;
    let recorded = text(
        required(root, "seal_sha256", label)?,
        &format!("{label}.seal_sha256"),
    )?;
    if !is_sha256(recorded) {
        return Err(format!("{label}.seal_sha256 must be SHA-256"));
    }
    let mut unsigned = root.clone();
    unsigned.remove("seal_sha256");
    let computed = canonical_sha256(&Value::Object(unsigned))?;
    if computed != recorded {
        return Err(format!(
            "{label} seal mismatch: recorded {recorded}, computed {computed}"
        ));
    }
    Ok(recorded.to_owned())
}

fn validate_source_config(config: &Value) -> Result<(), String> {
    let root = object(config, "source config")?;
    let architectures = array(
        required(root, "architectures", "source config")?,
        "source config.architectures",
    )?;
    if architectures.len() != 1
        || text(&architectures[0], "source config.architectures[0]")? != "Qwen3MoeForCausalLM"
    {
        return Err("source config must select only Qwen3MoeForCausalLM".into());
    }
    for (key, expected) in [
        ("model_type", "qwen3_moe"),
        ("hidden_act", "silu"),
        ("torch_dtype", "bfloat16"),
    ] {
        require_text(root, key, expected, "source config")?;
    }
    for (key, expected) in [
        ("hidden_size", HIDDEN_SIZE),
        ("vocab_size", VOCAB_SIZE),
        ("num_hidden_layers", LAYER_COUNT),
        ("num_attention_heads", ATTENTION_HEADS),
        ("num_key_value_heads", KV_HEADS),
        ("head_dim", HEAD_DIM),
        ("num_experts", EXPERT_COUNT),
        ("num_experts_per_tok", TOP_K),
        ("moe_intermediate_size", MOE_INTERMEDIATE),
        ("decoder_sparse_step", 1),
    ] {
        require_u64(root, key, expected, "source config")?;
    }
    for (key, expected) in [
        ("attention_bias", false),
        ("norm_topk_prob", true),
        ("use_cache", true),
        ("use_sliding_window", false),
        ("tie_word_embeddings", false),
    ] {
        require_bool(root, key, expected, "source config")?;
    }
    if required(root, "attention_dropout", "source config")?.as_f64() != Some(0.0) {
        return Err("source config.attention_dropout must be exactly 0.0".into());
    }
    if required(root, "rms_norm_eps", "source config")?.as_f64() != Some(1e-6) {
        return Err("source config.rms_norm_eps must be exactly 1e-6".into());
    }
    for key in ["sliding_window", "rope_scaling"] {
        if !required(root, key, "source config")?.is_null() {
            return Err(format!(
                "source config.{key} is an unsupported alternate semantics path"
            ));
        }
    }
    if !array(
        required(root, "mlp_only_layers", "source config")?,
        "source config.mlp_only_layers",
    )?
    .is_empty()
    {
        return Err(
            "source config.mlp_only_layers must be empty; dense MLP fallback is unsupported".into(),
        );
    }
    Ok(())
}

fn require_markers_in_order(text: &str, section: &str, markers: &[&str]) -> Result<(), String> {
    let mut next = 0usize;
    for marker in markers {
        let relative = text[next..]
            .find(marker)
            .ok_or_else(|| format!("source modeling code lacks {section} marker {marker:?}"))?;
        next = next
            .checked_add(relative)
            .and_then(|value| value.checked_add(marker.len()))
            .ok_or_else(|| format!("source modeling {section} marker offset overflow"))?;
    }
    Ok(())
}

fn validate_modeling_code(code: &str) -> Result<(), String> {
    for (section, markers) in [
        (
            "embedding/canonical causal model",
            &[
                "inputs_embeds = self.embed_tokens(input_ids)",
                "causal_mask = mask_function(",
                "hidden_states = inputs_embeds",
                "for decoder_layer in self.layers[: self.config.num_hidden_layers]:",
                "hidden_states = self.norm(hidden_states)",
                "logits = self.lm_head(hidden_states",
            ][..],
        ),
        (
            "RMSNorm scalar boundary",
            &[
                "hidden_states = hidden_states.to(torch.float32)",
                "variance = hidden_states.pow(2).mean(-1, keepdim=True)",
                "torch.rsqrt(variance + self.variance_epsilon)",
                "return self.weight * hidden_states.to(input_dtype)",
            ][..],
        ),
        (
            "attention projection/rope/cache/output",
            &[
                "query_states = self.q_norm(self.q_proj(hidden_states)",
                "key_states = self.k_norm(self.k_proj(hidden_states)",
                "value_states = self.v_proj(hidden_states)",
                "query_states, key_states = apply_rotary_pos_emb",
                "key_states, value_states = past_key_values.update",
                "attn_output, attn_weights = attention_interface(",
                "attn_output = self.o_proj(attn_output)",
            ][..],
        ),
        (
            "causal attention scalar order",
            &[
                "attn_weights = torch.matmul(query, key_states.transpose(2, 3)) * scaling",
                "attn_weights = attn_weights + attention_mask",
                "softmax(attn_weights, dim=-1, dtype=torch.float32).to(query.dtype)",
                "attn_output = torch.matmul(attn_weights, value_states)",
            ][..],
        ),
        (
            "router top-k and normalized source weights",
            &[
                "router_logits = F.linear(hidden_states, self.weight)",
                "router_probs = torch.nn.functional.softmax(router_logits, dtype=torch.float, dim=-1)",
                "torch.topk(router_probs, self.top_k, dim=-1)",
                "if self.norm_topk_prob:",
                "router_top_value /= router_top_value.sum(dim=-1, keepdim=True)",
                "router_top_value = router_top_value.to(router_logits.dtype)",
            ][..],
        ),
        (
            "selected expert gate/up/SwiGLU/down/combine",
            &[
                "gate, up = nn.functional.linear(current_state, self.gate_up_proj[expert_idx]).chunk(2, dim=-1)",
                "current_hidden_states = self.act_fn(gate) * up",
                "current_hidden_states = nn.functional.linear(current_hidden_states, self.down_proj[expert_idx])",
                "current_hidden_states = current_hidden_states * top_k_weights[token_idx, top_k_pos, None]",
                "final_hidden_states.index_add_(0, token_idx, current_hidden_states.to(final_hidden_states.dtype))",
            ][..],
        ),
        (
            "two residual additions",
            &[
                "residual = hidden_states",
                "hidden_states = self.input_layernorm(hidden_states)",
                "hidden_states = residual + hidden_states",
                "residual = hidden_states",
                "hidden_states = self.post_attention_layernorm(hidden_states)",
                "hidden_states = self.mlp(hidden_states)",
                "hidden_states = residual + hidden_states",
            ][..],
        ),
    ] {
        require_markers_in_order(code, section, markers)?;
    }
    for marker in [
        "self.is_causal = True",
        "self.num_key_value_groups = config.num_attention_heads // config.num_key_value_heads",
        "self.act_fn = ACT2FN[config.hidden_act]",
        "past_key_values = DynamicCache(config=self.config)",
    ] {
        if !code.contains(marker) {
            return Err(format!(
                "source modeling code lacks required marker {marker:?}"
            ));
        }
    }
    Ok(())
}

fn validate_range_authority(document: &Value, pins: Pins<'_>) -> Result<(), String> {
    let root = object(document, "range authority document")?;
    let authority = required(root, "authority", "range authority document")?;
    let recorded_content_hash = text(
        required(root, "authority_content_sha256", "range authority document")?,
        "range authority document.authority_content_sha256",
    )?;
    if !is_sha256(recorded_content_hash) {
        return Err("range authority content hash must be SHA-256".into());
    }
    let computed_content_hash = canonical_sha256(authority)?;
    if computed_content_hash != recorded_content_hash {
        return Err("range authority content hash does not bind its authority material".into());
    }
    if computed_content_hash != pins.range_authority_content_sha256 {
        return Err(
            "range authority content hash drifted from this attester's pinned source authority"
                .into(),
        );
    }
    let authority = object(authority, "range authority")?;
    require_text(
        authority,
        "schema",
        RANGE_AUTHORITY_SCHEMA,
        "range authority",
    )?;
    require_text(
        authority,
        "status",
        RANGE_AUTHORITY_STATUS,
        "range authority",
    )?;
    let source = object(
        required(authority, "source", "range authority")?,
        "range authority.source",
    )?;
    require_text(
        source,
        "model_id",
        SOURCE_MODEL_ID,
        "range authority.source",
    )?;
    require_text(
        source,
        "source_revision",
        pins.source_revision,
        "range authority.source",
    )?;
    require_u64(
        source,
        "source_tensor_count",
        SOURCE_TENSOR_COUNT,
        "range authority.source",
    )?;
    require_u64(
        source,
        "source_shard_count",
        SOURCE_SHARD_COUNT,
        "range authority.source",
    )?;
    let source_index = object(
        required(source, "source_index", "range authority.source")?,
        "range authority.source.source_index",
    )?;
    require_text(
        source_index,
        "sha256",
        pins.source_index_sha256,
        "range authority.source.source_index",
    )?;
    let scope = object(
        required(authority, "exact_streamed_oracle_scope", "range authority")?,
        "range authority.exact_streamed_oracle_scope",
    )?;
    for (key, expected) in [
        ("source_template_token_count", PREFIX_TOKEN_COUNT),
        ("forced_identical_continuation_token_id", FORCED_TOKEN_ID),
        ("total_forwards_per_replay_arm", FORWARDS_PER_ARM),
        ("layers", LAYER_COUNT),
        ("top_k_routes_per_token", TOP_K),
    ] {
        require_u64(
            scope,
            key,
            expected,
            "range authority.exact_streamed_oracle_scope",
        )?;
    }
    for key in [
        "sampling_or_autoregressive_feedback_forbidden",
        "all_128_expert_tensor_names_are_pre-authorized_but_payload_reads_must_still_be_route_selected",
    ] {
        require_bool(scope, key, true, "range authority.exact_streamed_oracle_scope")?;
    }
    let boundary = object(
        required(authority, "metadata_access_boundary", "range authority")?,
        "range authority.metadata_access_boundary",
    )?;
    require_u64(
        boundary,
        "source_tensor_payload_bytes_read",
        0,
        "range authority.metadata_access_boundary",
    )?;
    for key in [
        "tensor_payload_hashes_collected",
        "whole_shard_payload_checksum_collected",
        "mmap_or_memory_map_used",
        "source_model_instantiated",
        "gpu_or_metal_invoked",
        "server_started",
        "hcli_invoked",
        "lease_requested",
    ] {
        require_bool(
            boundary,
            key,
            false,
            "range authority.metadata_access_boundary",
        )?;
    }
    let tensor_roles = array(
        required(authority, "tensors", "range authority")?,
        "range authority.tensors",
    )?
    .iter()
    .map(|tensor| {
        let tensor = object(tensor, "range authority tensor")?;
        text(
            required(tensor, "role", "range authority tensor")?,
            "range authority tensor.role",
        )
        .map(str::to_owned)
    })
    .collect::<Result<BTreeSet<_>, _>>()?;
    for role in [
        "embedding_row",
        "input_rmsnorm",
        "q_rmsnorm",
        "k_rmsnorm",
        "q_projection",
        "k_projection",
        "v_projection",
        "o_projection",
        "post_attention_rmsnorm",
        "router_all_128_logits",
        "selected_expert_gate_projection",
        "selected_expert_up_projection",
        "selected_expert_down_projection",
        "final_rmsnorm",
        "lm_head_all_rows",
    ] {
        if !tensor_roles.contains(role) {
            return Err(format!(
                "range authority is missing source tensor role {role:?}"
            ));
        }
    }
    Ok(())
}

fn validate_trace_contract(document: &Value, pins: Pins<'_>) -> Result<(), String> {
    let recorded_seal = verify_seal(document, "sealed source replay contract")?;
    if recorded_seal != pins.trace_contract_seal_sha256 {
        return Err("sealed source replay contract seal drifted from the pinned contract".into());
    }
    let root = object(document, "sealed source replay contract")?;
    require_text(
        root,
        "schema",
        TRACE_CONTRACT_SCHEMA,
        "sealed source replay contract",
    )?;
    require_text(
        root,
        "status",
        TRACE_CONTRACT_STATUS,
        "sealed source replay contract",
    )?;
    let exact = object(
        required(root, "exact_input", "sealed source replay contract")?,
        "sealed source replay contract.exact_input",
    )?;
    require_u64(
        exact,
        "source_template_token_count",
        PREFIX_TOKEN_COUNT,
        "sealed source replay contract.exact_input",
    )?;
    require_u64(
        exact,
        "forced_identical_continuation_token_id",
        FORCED_TOKEN_ID,
        "sealed source replay contract.exact_input",
    )?;
    require_text(
        exact,
        "source_template_token_ids_u32le_sha256",
        pins.template_token_ids_u32le_sha256,
        "sealed source replay contract.exact_input",
    )?;
    for key in [
        "sampling_or_autoregressive_feedback_is_forbidden",
        "source_must_execute_the_same_369_token_prefix_then_the_forced_token",
    ] {
        require_bool(
            exact,
            key,
            true,
            "sealed source replay contract.exact_input",
        )?;
    }
    let evidence = object(
        required(root, "evidence", "sealed source replay contract")?,
        "sealed source replay contract.evidence",
    )?;
    require_text(
        evidence,
        "source_revision",
        pins.source_revision,
        "sealed source replay contract.evidence",
    )?;
    require_u64(
        evidence,
        "source_shard_count",
        SOURCE_SHARD_COUNT,
        "sealed source replay contract.evidence",
    )?;
    let config = object(
        required(
            evidence,
            "source_config",
            "sealed source replay contract.evidence",
        )?,
        "sealed source replay contract.evidence.source_config",
    )?;
    require_text(
        config,
        "sha256",
        pins.source_config_sha256,
        "sealed source replay contract.evidence.source_config",
    )?;
    let index = object(
        required(
            evidence,
            "source_index",
            "sealed source replay contract.evidence",
        )?,
        "sealed source replay contract.evidence.source_index",
    )?;
    require_text(
        index,
        "sha256",
        pins.source_index_sha256,
        "sealed source replay contract.evidence.source_index",
    )?;
    Ok(())
}

fn prepared_document(
    args: &Args,
    config_sha256: &str,
    modeling_sha256: &str,
    range_authority_sha256: &str,
    trace_contract_sha256: &str,
) -> Value {
    let traversals_per_arm = FORWARDS_PER_ARM * LAYER_COUNT;
    json!({
        "schema": RESULT_SCHEMA,
        "status": RESULT_STATUS,
        "claim_boundary": "Prepared metadata-only source operator/accumulation semantics contract only. No source tensor payload was opened, no source model was instantiated, no numerical source inference or quality comparison ran, and no coherence, HCLI, TPS, TG, tournament, serving, or promotion result exists.",
        "execution_boundary": {
            "source_tensor_payload_opened": false,
            "source_safetensors_or_other_weight_path_accepted": false,
            "source_model_instantiated": false,
            "source_inference_executed": false,
            "gpu_or_metal_invoked": false,
            "server_started": false,
            "hcli_invoked": false,
            "lease_requested": false,
            "source_quality_or_coherence_claim_made": false,
            "tps_or_tg_claim_made": false
        },
        "pinned_source_binding": {
            "source_model_id": SOURCE_MODEL_ID,
            "source_revision": args.source_revision,
            "source_config": {"path": args.source_config, "sha256": config_sha256},
            "source_modeling_code": {"path": args.modeling_code, "sha256": modeling_sha256},
            "source_index_sha256": SOURCE_INDEX_SHA256,
            "geometry": {
                "layers": LAYER_COUNT,
                "hidden_size": HIDDEN_SIZE,
                "vocab_size": VOCAB_SIZE,
                "attention_heads": ATTENTION_HEADS,
                "key_value_heads": KV_HEADS,
                "head_dim": HEAD_DIM,
                "experts": EXPERT_COUNT,
                "top_k": TOP_K,
                "moe_intermediate": MOE_INTERMEDIATE,
                "source_tensor_count": SOURCE_TENSOR_COUNT,
                "source_shard_count": SOURCE_SHARD_COUNT
            }
        },
        "consumed_metadata_contracts": {
            "range_authority": {
                "path": args.range_authority,
                "document_sha256": range_authority_sha256,
                "authority_content_sha256": RANGE_AUTHORITY_CONTENT_SHA256,
                "source_payload_read_by_this_attester": false
            },
            "sealed_replay_contract": {
                "path": args.trace_contract,
                "document_sha256": trace_contract_sha256,
                "seal_sha256": TRACE_CONTRACT_SEAL_SHA256,
                "source_payload_read_by_this_attester": false
            }
        },
        "metadata_checked_source_code_surface": {
            "source_code_marker_order_verified": true,
            "selected_reference_profile": "pinned_transformers_qwen3_moe_eager_reference_semantics_only",
            "decorator_or_runtime_kernel_substitution_is_not_execution_evidence": true,
            "selectable_attention_backend_requires_fresh_exact_execution_attestation": true,
            "automatic_cast_or_backend_precision_change_requires_fresh_exact_execution_attestation": true
        },
        "exact_scalar_and_order_requirements": {
            "embedding": {
                "required_sequence": ["exact_u32_token_id_from_sealed_trace", "one_source_embedding_row_for_that_token", "row_major_increasing_hidden_index_read", "record_actual_source_dtype_and_store_boundary"],
                "tokenization_or_retokenization_substitution_forbidden": true,
                "embedding_row_must_be_bound_to_the_range_authority": true
            },
            "rmsnorm": {
                "required_sequence": ["source_input_dtype_capture", "convert_hidden_states_to_f32", "mean_of_squared_hidden_values", "add_rms_norm_eps_1e_minus_6", "rsqrt", "multiply_norm_weight", "cast_to_source_input_dtype"],
                "all_f32_accumulation_and_cast_store_boundaries_must_be_recorded": true,
                "fused_or_approximate_norm_without_source_equivalence_evidence_forbidden": true
            },
            "attention_and_kv_causal_order": {
                "required_sequence": ["input_rmsnorm", "q_projection_then_q_rmsnorm", "k_projection_then_k_rmsnorm", "v_projection_without_kv_norm", "rope_on_q_and_k", "append_rotated_k_and_v_to_layer_kv_cache", "construct_or_apply_causal_mask_including_current_token", "repeat_kv_by_group_factor_8", "qk_dot_plus_additive_mask", "f32_softmax_then_cast_to_query_dtype", "attention_value_dot", "o_projection", "first_residual_add"],
                "kv_append_must_happen_before_any_causal_read": true,
                "future_keys_or_values_must_not_be_visible": true,
                "linear_dot_k_order_and_accumulator_store_boundaries_must_be_recorded": true,
                "attention_backend_must_be_explicitly_attested_as_eager_reference_or_equivalent": true
            },
            "router_topk_and_weights": {
                "required_sequence": ["post_attention_rmsnorm", "router_linear_all_128_logits", "f32_softmax_over_all_128", "topk_8_on_router_probabilities", "normalize_selected_topk_probability_sum", "cast_selected_weights_to_router_logits_dtype"],
                "exact_selected_expert_ids_weights_and_topk_tie_behavior_must_be_recorded": true,
                "topk_on_logits_or_unormalized_selected_weights_forbidden": true
            },
            "selected_expert_gate_up_swiglu_down": {
                "required_sequence": ["source_selected_expert_and_token_index_order", "gate_and_up_projection_from_the_same_current_state", "silu_gate_times_up", "down_projection", "multiply_by_that_selected_route_weight", "source_index_add_combine_order"],
                "all_eight_selected_routes_must_be_executed_once_per_sparse_layer_token": true,
                "route_slot_or_expert_iteration_order_must_be_recorded_not_invented": true,
                "parallel_reduction_or_fused_moe_without_source_order_evidence_forbidden": true
            },
            "residuals": {
                "required_sequence": ["first_residual_equals_pre_attention_residual_plus_attention_output", "second_residual_equals_post_attention_residual_plus_moe_output"],
                "addition_dtype_accumulation_and_store_boundary_must_be_recorded": true,
                "residual_reassociation_forbidden": true
            },
            "final_norm_and_head": {
                "required_sequence": ["final_rmsnorm_with_the_same_f32_rule", "lm_head_all_vocabulary_rows_in_increasing_k_order", "record_source_logit_dtype", "explicit_capture_widening_to_full_f32_if_needed", "retain_full_f32_logit_vector"],
                "vocab_rows": VOCAB_SIZE,
                "retained_f32_logit_bytes_per_endpoint": F32_FINAL_LOGIT_BYTES,
                "topk_only_or_argmax_only_logit_retention_forbidden": true
            },
            "sealed_369_prefix_plus_forced_two_arm_replay": {
                "source_template_token_count": PREFIX_TOKEN_COUNT,
                "forced_identical_continuation_token_id": FORCED_TOKEN_ID,
                "forwards_per_replay_arm": FORWARDS_PER_ARM,
                "layers_per_forward": LAYER_COUNT,
                "layer_traversals_per_replay_arm": traversals_per_arm,
                "two_native_control_candidate_arms_layer_traversals": traversals_per_arm * 2,
                "same_prefix_then_same_forced_token_required_for_every_aligned_arm": true,
                "sampling_autoregressive_feedback_or_candidate_dependent_token_selection_forbidden": true,
                "required_retained_endpoint_vectors": {
                    "source_prefix_and_forced": 2,
                    "native_control_prefix_and_forced": 2,
                    "native_candidate_prefix_and_forced": 2,
                    "total": 6
                }
            }
        },
        "unsupported_or_drift_rejections": [
            "source revision, config hash, modeling-code hash, range-authority hash, replay-contract seal, source-index hash, or source-template-token hash drift",
            "any source weight payload path, safetensors shard, mmap, model instantiation, GPU/Metal, server, HCLI, lease, sampling, or generation request supplied to this attester",
            "non-BF16 source weights, attention bias, attention dropout, sliding-window attention, RoPE scaling, disabled cache, tied embedding/head, dense MLP-only layers, non-SiLU activation, non-8 top-k, unnormalized top-k probabilities, or non-48-layer geometry",
            "flash, SDPA, flex, hub-kernel, fused, approximate, reordered, parallel-reduced, or autocast semantics unless a later execution attestation proves source-equivalent scalar/order behavior",
            "KV read before append, future-token visibility, top-k logits in place of probabilities, route-order invention, residual reassociation, or partial-logit retention"
        ],
        "future_exact_execution_attestation": {
            "schema": FUTURE_ATTESTATION_SCHEMA,
            "status_only_after_real_separately_leased_source_execution": FUTURE_ATTESTATION_STATUS,
            "must_rebind_all_pinned_source_and_contract_hashes": true,
            "must_record_per_operator_scalar_dtype_accumulator_and_store_boundaries": true,
            "must_record_per_layer_kv_append_and_causal_read_order": true,
            "must_record_each_router_logit_topk_id_normalized_weight_and_expert_combine_event": true,
            "must_record_all_48_layers_for_369_prefix_and_forced_token": true,
            "must_retain_and_hash_six_full_f32_endpoint_logit_vectors_before_any_three_way_scoring": true,
            "fresh_execution_evidence_required": [
                "revalidated sealed range-authority and replay-contract bindings",
                "actual source code/config hashes and selected attention/backend profile",
                "bounded source-reader raw BF16 row-order evidence without full-shard mapping or caching",
                "operator event trace containing scalar dtype, k-order, accumulation and store boundaries",
                "KV cache append/read and causal-mask evidence for every layer and forward",
                "selected top-8 IDs/weights plus exact source expert/index-add combine order",
                "full F32 prefix and forced-endpoint vectors and hashes for source, control, and candidate",
                "exclusive non-timed source-oracle lease plus zero-swap observations"
            ]
        }
    })
}

fn build_document(args: &Args, pins: Pins<'_>) -> Result<Value, String> {
    if args.source_revision != pins.source_revision {
        return Err(
            "requested source revision differs from this attester's pinned source revision".into(),
        );
    }
    require_extension(&args.range_authority, "json", "range authority")?;
    require_extension(
        &args.trace_contract,
        "json",
        "sealed source replay contract",
    )?;
    require_extension(&args.out, "json", "output")?;
    let config_bytes = read_regular_metadata_file(
        &args.source_config,
        "source config",
        MAX_CONFIG_BYTES,
        Some("config.json"),
    )?;
    let config_sha256 = sha256_hex(&config_bytes);
    if config_sha256 != pins.source_config_sha256 {
        return Err("source config checksum drifted from the pinned source config".into());
    }
    validate_source_config(&parse_json(&config_bytes, "source config")?)?;

    let modeling_bytes = read_regular_metadata_file(
        &args.modeling_code,
        "source modeling code",
        MAX_MODELING_CODE_BYTES,
        Some("modeling_qwen3_moe.py"),
    )?;
    let modeling_sha256 = sha256_hex(&modeling_bytes);
    if modeling_sha256 != pins.source_modeling_sha256 {
        return Err("source modeling-code checksum drifted from the pinned source code".into());
    }
    let modeling_text = std::str::from_utf8(&modeling_bytes)
        .map_err(|error| format!("source modeling code is not UTF-8: {error}"))?;
    validate_modeling_code(modeling_text)?;

    let range_bytes = read_regular_metadata_file(
        &args.range_authority,
        "range authority",
        MAX_RANGE_AUTHORITY_BYTES,
        None,
    )?;
    let range_sha256 = sha256_hex(&range_bytes);
    validate_range_authority(&parse_json(&range_bytes, "range authority")?, pins)?;

    let trace_bytes = read_regular_metadata_file(
        &args.trace_contract,
        "sealed source replay contract",
        MAX_TRACE_CONTRACT_BYTES,
        None,
    )?;
    let trace_sha256 = sha256_hex(&trace_bytes);
    validate_trace_contract(
        &parse_json(&trace_bytes, "sealed source replay contract")?,
        pins,
    )?;

    Ok(prepared_document(
        args,
        &config_sha256,
        &modeling_sha256,
        &range_sha256,
        &trace_sha256,
    ))
}

fn write_new_json(path: &Path, value: &Value) -> Result<(), String> {
    if !path.is_absolute() {
        return Err("output path must be absolute".into());
    }
    let parent = path
        .parent()
        .ok_or_else(|| "output path must have a parent".to_owned())?;
    let parent_metadata = fs::symlink_metadata(parent)
        .map_err(|error| format!("cannot stat output parent {}: {error}", parent.display()))?;
    if parent_metadata.file_type().is_symlink() || !parent_metadata.file_type().is_dir() {
        return Err("output parent must be a real directory, not a symlink".into());
    }
    if fs::symlink_metadata(path).is_ok() {
        return Err(format!("output already exists: {}", path.display()));
    }
    let bytes = serde_json::to_vec_pretty(value)
        .map_err(|error| format!("cannot serialize prepared semantics contract: {error}"))?;
    let mut file = OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(path)
        .map_err(|error| format!("cannot create output {}: {error}", path.display()))?;
    file.write_all(&bytes)
        .and_then(|_| file.write_all(b"\n"))
        .and_then(|_| file.sync_all())
        .map_err(|error| format!("cannot write output {}: {error}", path.display()))
}

fn run(args: Args) -> Result<Value, String> {
    let document = build_document(&args, PRODUCTION_PINS)?;
    write_new_json(&args.out, &document)?;
    Ok(document)
}

fn main() {
    match parse_args().and_then(run) {
        Ok(document) => match serde_json::to_string_pretty(&document) {
            Ok(text) => println!("{text}"),
            Err(error) => {
                eprintln!("cannot serialize prepared semantics contract: {error}");
                process::exit(1);
            }
        },
        Err(error) => {
            eprintln!("{error}");
            process::exit(2);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::TempDir;

    fn fixture_config() -> Value {
        json!({
            "architectures": ["Qwen3MoeForCausalLM"],
            "model_type": "qwen3_moe",
            "hidden_act": "silu",
            "torch_dtype": "bfloat16",
            "hidden_size": HIDDEN_SIZE,
            "vocab_size": VOCAB_SIZE,
            "num_hidden_layers": LAYER_COUNT,
            "num_attention_heads": ATTENTION_HEADS,
            "num_key_value_heads": KV_HEADS,
            "head_dim": HEAD_DIM,
            "num_experts": EXPERT_COUNT,
            "num_experts_per_tok": TOP_K,
            "moe_intermediate_size": MOE_INTERMEDIATE,
            "decoder_sparse_step": 1,
            "attention_bias": false,
            "norm_topk_prob": true,
            "use_cache": true,
            "use_sliding_window": false,
            "tie_word_embeddings": false,
            "attention_dropout": 0.0,
            "rms_norm_eps": 1e-6,
            "sliding_window": null,
            "rope_scaling": null,
            "mlp_only_layers": []
        })
    }

    fn fixture_modeling_code() -> String {
        // This is deliberately a synthetic marker fixture, not a source model
        // and not executable Python.  It contains only the source operations
        // the metadata parser needs to order-check.
        [
            "self.is_causal = True",
            "self.num_key_value_groups = config.num_attention_heads // config.num_key_value_heads",
            "self.act_fn = ACT2FN[config.hidden_act]",
            "hidden_states = hidden_states.to(torch.float32)",
            "variance = hidden_states.pow(2).mean(-1, keepdim=True)",
            "torch.rsqrt(variance + self.variance_epsilon)",
            "return self.weight * hidden_states.to(input_dtype)",
            "attn_weights = torch.matmul(query, key_states.transpose(2, 3)) * scaling",
            "attn_weights = attn_weights + attention_mask",
            "softmax(attn_weights, dim=-1, dtype=torch.float32).to(query.dtype)",
            "attn_output = torch.matmul(attn_weights, value_states)",
            "query_states = self.q_norm(self.q_proj(hidden_states)",
            "key_states = self.k_norm(self.k_proj(hidden_states)",
            "value_states = self.v_proj(hidden_states)",
            "query_states, key_states = apply_rotary_pos_emb",
            "key_states, value_states = past_key_values.update",
            "attn_output, attn_weights = attention_interface(",
            "attn_output = self.o_proj(attn_output)",
            "router_logits = F.linear(hidden_states, self.weight)",
            "router_probs = torch.nn.functional.softmax(router_logits, dtype=torch.float, dim=-1)",
            "torch.topk(router_probs, self.top_k, dim=-1)",
            "if self.norm_topk_prob:",
            "router_top_value /= router_top_value.sum(dim=-1, keepdim=True)",
            "router_top_value = router_top_value.to(router_logits.dtype)",
            "gate, up = nn.functional.linear(current_state, self.gate_up_proj[expert_idx]).chunk(2, dim=-1)",
            "current_hidden_states = self.act_fn(gate) * up",
            "current_hidden_states = nn.functional.linear(current_hidden_states, self.down_proj[expert_idx])",
            "current_hidden_states = current_hidden_states * top_k_weights[token_idx, top_k_pos, None]",
            "final_hidden_states.index_add_(0, token_idx, current_hidden_states.to(final_hidden_states.dtype))",
            "residual = hidden_states",
            "hidden_states = self.input_layernorm(hidden_states)",
            "hidden_states = residual + hidden_states",
            "residual = hidden_states",
            "hidden_states = self.post_attention_layernorm(hidden_states)",
            "hidden_states = self.mlp(hidden_states)",
            "hidden_states = residual + hidden_states",
            "past_key_values = DynamicCache(config=self.config)",
            "inputs_embeds = self.embed_tokens(input_ids)",
            "causal_mask = mask_function(",
            "hidden_states = inputs_embeds",
            "for decoder_layer in self.layers[: self.config.num_hidden_layers]:",
            "hidden_states = self.norm(hidden_states)",
            "logits = self.lm_head(hidden_states",
        ]
        .join("\n")
    }

    fn fixture_range_authority(revision: &str, index_sha256: &str) -> Value {
        let authority = json!({
            "schema": RANGE_AUTHORITY_SCHEMA,
            "status": RANGE_AUTHORITY_STATUS,
            "source": {
                "model_id": SOURCE_MODEL_ID,
                "source_revision": revision,
                "source_tensor_count": SOURCE_TENSOR_COUNT,
                "source_shard_count": SOURCE_SHARD_COUNT,
                "source_index": {"sha256": index_sha256}
            },
            "exact_streamed_oracle_scope": {
                "source_template_token_count": PREFIX_TOKEN_COUNT,
                "forced_identical_continuation_token_id": FORCED_TOKEN_ID,
                "total_forwards_per_replay_arm": FORWARDS_PER_ARM,
                "layers": LAYER_COUNT,
                "top_k_routes_per_token": TOP_K,
                "sampling_or_autoregressive_feedback_forbidden": true,
                "all_128_expert_tensor_names_are_pre-authorized_but_payload_reads_must_still_be_route_selected": true
            },
            "metadata_access_boundary": {
                "source_tensor_payload_bytes_read": 0,
                "tensor_payload_hashes_collected": false,
                "whole_shard_payload_checksum_collected": false,
                "mmap_or_memory_map_used": false,
                "source_model_instantiated": false,
                "gpu_or_metal_invoked": false,
                "server_started": false,
                "hcli_invoked": false,
                "lease_requested": false
            },
            "tensors": [
                {"role": "embedding_row"}, {"role": "input_rmsnorm"}, {"role": "q_rmsnorm"},
                {"role": "k_rmsnorm"}, {"role": "q_projection"}, {"role": "k_projection"},
                {"role": "v_projection"}, {"role": "o_projection"}, {"role": "post_attention_rmsnorm"},
                {"role": "router_all_128_logits"}, {"role": "selected_expert_gate_projection"},
                {"role": "selected_expert_up_projection"}, {"role": "selected_expert_down_projection"},
                {"role": "final_rmsnorm"}, {"role": "lm_head_all_rows"}
            ]
        });
        let content = canonical_sha256(&authority).expect("canonical fixture authority");
        json!({"authority_content_sha256": content, "authority": authority})
    }

    fn sealed_fixture_trace(
        revision: &str,
        config_sha256: &str,
        index_sha256: &str,
        token_sha256: &str,
    ) -> Value {
        let unsigned = json!({
            "schema": TRACE_CONTRACT_SCHEMA,
            "status": TRACE_CONTRACT_STATUS,
            "exact_input": {
                "source_template_token_count": PREFIX_TOKEN_COUNT,
                "forced_identical_continuation_token_id": FORCED_TOKEN_ID,
                "source_template_token_ids_u32le_sha256": token_sha256,
                "sampling_or_autoregressive_feedback_is_forbidden": true,
                "source_must_execute_the_same_369_token_prefix_then_the_forced_token": true
            },
            "evidence": {
                "source_revision": revision,
                "source_shard_count": SOURCE_SHARD_COUNT,
                "source_config": {"sha256": config_sha256},
                "source_index": {"sha256": index_sha256}
            }
        });
        let seal = canonical_sha256(&unsigned).expect("canonical fixture trace");
        let mut root = unsigned.as_object().expect("fixture object").clone();
        root.insert("seal_sha256".into(), Value::String(seal));
        Value::Object(root)
    }

    fn write_json(path: &Path, value: &Value) {
        fs::write(
            path,
            serde_json::to_vec(value).expect("serialize fixture JSON"),
        )
        .expect("write fixture JSON");
    }

    fn fixture_args_and_pins() -> (TempDir, Args, Pins<'static>) {
        let temp = TempDir::new().expect("temporary fixture directory");
        let config_path = temp.path().join("config.json");
        let code_path = temp.path().join("modeling_qwen3_moe.py");
        let range_path = temp.path().join("range-authority.json");
        let trace_path = temp.path().join("trace-contract.json");
        let out_path = temp.path().join("result.json");
        let config = fixture_config();
        let code = fixture_modeling_code();
        write_json(&config_path, &config);
        fs::write(&code_path, &code).expect("write fixture source snippet");
        let config_sha256 = sha256_hex(&serde_json::to_vec(&config).expect("serialize config"));
        let code_sha256 = sha256_hex(code.as_bytes());
        let revision = "a".repeat(40);
        let index_sha256 = "b".repeat(64);
        let token_sha256 = "c".repeat(64);
        let range = fixture_range_authority(&revision, &index_sha256);
        let range_hash = text(
            range.get("authority_content_sha256").expect("range hash"),
            "range authority content hash",
        )
        .expect("range content hash")
        .to_owned();
        write_json(&range_path, &range);
        let trace = sealed_fixture_trace(&revision, &config_sha256, &index_sha256, &token_sha256);
        let trace_seal = text(trace.get("seal_sha256").expect("trace seal"), "trace seal")
            .expect("trace seal text")
            .to_owned();
        write_json(&trace_path, &trace);
        let pins = Pins {
            source_revision: Box::leak(revision.into_boxed_str()),
            source_config_sha256: Box::leak(config_sha256.into_boxed_str()),
            source_modeling_sha256: Box::leak(code_sha256.into_boxed_str()),
            range_authority_content_sha256: Box::leak(range_hash.into_boxed_str()),
            trace_contract_seal_sha256: Box::leak(trace_seal.into_boxed_str()),
            source_index_sha256: Box::leak(index_sha256.into_boxed_str()),
            template_token_ids_u32le_sha256: Box::leak(token_sha256.into_boxed_str()),
        };
        let args = Args {
            source_config: config_path,
            modeling_code: code_path,
            range_authority: range_path,
            trace_contract: trace_path,
            source_revision: pins.source_revision.to_owned(),
            out: out_path,
        };
        (temp, args, pins)
    }

    #[test]
    fn synthetic_metadata_fixture_emits_prepared_contract_with_all_operator_families() {
        let (_temp, args, pins) = fixture_args_and_pins();
        let document = build_document(&args, pins).expect("valid synthetic metadata fixture");
        assert_eq!(document["schema"], RESULT_SCHEMA);
        assert_eq!(document["status"], RESULT_STATUS);
        assert_eq!(
            document["execution_boundary"]["source_tensor_payload_opened"],
            false
        );
        assert_eq!(
            document["execution_boundary"]["gpu_or_metal_invoked"],
            false
        );
        assert_eq!(
            document["exact_scalar_and_order_requirements"]["embedding"]
                ["tokenization_or_retokenization_substitution_forbidden"],
            true
        );
        assert_eq!(
            document["exact_scalar_and_order_requirements"]["attention_and_kv_causal_order"]
                ["kv_append_must_happen_before_any_causal_read"],
            true
        );
        assert_eq!(
            document["exact_scalar_and_order_requirements"]["router_topk_and_weights"]
                ["exact_selected_expert_ids_weights_and_topk_tie_behavior_must_be_recorded"],
            true
        );
        assert_eq!(
            document["exact_scalar_and_order_requirements"]["selected_expert_gate_up_swiglu_down"]
                ["all_eight_selected_routes_must_be_executed_once_per_sparse_layer_token"],
            true
        );
        assert_eq!(
            document["exact_scalar_and_order_requirements"]["final_norm_and_head"]
                ["retained_f32_logit_bytes_per_endpoint"],
            F32_FINAL_LOGIT_BYTES
        );
        assert_eq!(
            document["exact_scalar_and_order_requirements"]
                ["sealed_369_prefix_plus_forced_two_arm_replay"]
                ["two_native_control_candidate_arms_layer_traversals"],
            FORWARDS_PER_ARM * LAYER_COUNT * 2
        );
    }

    #[test]
    fn synthetic_fixtures_reject_config_code_revision_and_seal_drift() {
        let (_temp, args, pins) = fixture_args_and_pins();
        let mut bad_config = fixture_config();
        bad_config["num_experts_per_tok"] = json!(7);
        write_json(&args.source_config, &bad_config);
        assert!(build_document(&args, pins)
            .unwrap_err()
            .contains("checksum drifted"));

        let (_temp, args, pins) = fixture_args_and_pins();
        let bad_code = fixture_modeling_code().replace(
            "key_states, value_states = past_key_values.update",
            "cache_update_removed",
        );
        fs::write(&args.modeling_code, bad_code).expect("rewrite synthetic source snippet");
        assert!(build_document(&args, pins)
            .unwrap_err()
            .contains("checksum drifted"));

        let (_temp, args, pins) = fixture_args_and_pins();
        let mut range = parse_json(
            &read_regular_metadata_file(
                &args.range_authority,
                "fixture range",
                MAX_RANGE_AUTHORITY_BYTES,
                None,
            )
            .expect("read range"),
            "fixture range",
        )
        .expect("parse range");
        range["authority"]["source"]["source_revision"] = json!("d".repeat(40));
        write_json(&args.range_authority, &range);
        assert!(build_document(&args, pins)
            .unwrap_err()
            .contains("content hash"));

        let (_temp, args, pins) = fixture_args_and_pins();
        let mut trace = parse_json(
            &read_regular_metadata_file(
                &args.trace_contract,
                "fixture trace",
                MAX_TRACE_CONTRACT_BYTES,
                None,
            )
            .expect("read trace"),
            "fixture trace",
        )
        .expect("parse trace");
        trace["exact_input"]["forced_identical_continuation_token_id"] = json!(950);
        write_json(&args.trace_contract, &trace);
        assert!(build_document(&args, pins)
            .unwrap_err()
            .contains("seal mismatch"));
    }

    #[test]
    fn synthetic_source_snippets_reject_unsupported_config_and_missing_operator_order() {
        let mut unsupported = fixture_config();
        unsupported["num_experts_per_tok"] = json!(7);
        assert!(validate_source_config(&unsupported)
            .unwrap_err()
            .contains("num_experts_per_tok"));

        let invalid_code = fixture_modeling_code().replace(
            "key_states, value_states = past_key_values.update",
            "cache_update_removed",
        );
        assert!(validate_modeling_code(&invalid_code)
            .unwrap_err()
            .contains("past_key_values.update"));

        let (_temp, mut args, pins) = fixture_args_and_pins();
        args.range_authority.set_extension("safetensors");
        assert!(build_document(&args, pins)
            .unwrap_err()
            .contains("payload paths are not accepted"));
    }

    #[test]
    fn parser_requires_absolute_paths_and_output_is_create_new() {
        let args = parse_args_from(vec![
            "--source-config".to_owned(),
            "/tmp/model.safetensors".to_owned(),
            "--modeling-code".to_owned(),
            "/tmp/modeling_qwen3_moe.py".to_owned(),
            "--range-authority".to_owned(),
            "/tmp/range.json".to_owned(),
            "--trace-contract".to_owned(),
            "/tmp/trace.json".to_owned(),
            "--source-revision".to_owned(),
            SOURCE_REVISION.to_owned(),
            "--out".to_owned(),
            "/tmp/out.json".to_owned(),
        ])
        .expect("paths are syntactically parsed");
        assert_eq!(
            args.source_config
                .file_name()
                .and_then(|item| item.to_str()),
            Some("model.safetensors")
        );
        let (_temp, args, pins) = fixture_args_and_pins();
        let document = build_document(&args, pins).expect("fixture output");
        write_new_json(&args.out, &document).expect("first create-new write");
        assert!(write_new_json(&args.out, &document)
            .unwrap_err()
            .contains("already exists"));
    }
}

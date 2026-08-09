//! CPU-only Qwen3-Coder-Next source-template A/B authority preflight.
//!
//! This target is deliberately a prompt-contract preflight, not a model
//! execution path.  It reads only five explicitly named, small source
//! sidecars: `config.json`, `tokenizer.json`, `tokenizer_config.json`,
//! `chat_template.jinja`, and `generation_config.json`.  Each is pinned by
//! SHA-256 to the admitted Qwen3-Coder-Next revision.  It never opens a
//! safetensors shard or Gravity payload, scans an artifact directory, creates
//! a Metal object, starts a model/server/HCLI process, invokes a decoder, or
//! measures generation, coherence, TPS, or TG.
//!
//! The bounded branch is intentionally narrow: exactly one `user` message,
//! no system prompt, no tools, no template delimiters in user content, and an
//! explicit 16-token future completion limit.  The output binds two distinct
//! source-template prompt controls suitable as *inputs* to a future coherent
//! generation/HCLI receipt.  It does not assert anything about future output.
//!
//! Example:
//! ```text
//! cargo run -p hawking-core --example ascension_qwen80_source_template_ab_authority_preflight -- \
//!   --source-dir /absolute/path/Qwen3-Coder-Next \
//!   --out /absolute/path/QWEN80_SOURCE_TEMPLATE_AB_AUTHORITY.json
//!
//! # Recompute and require exact agreement with a prior preflight:
//! cargo run -p hawking-core --example ascension_qwen80_source_template_ab_authority_preflight -- \
//!   --source-dir /absolute/path/Qwen3-Coder-Next \
//!   --revalidate /absolute/path/QWEN80_SOURCE_TEMPLATE_AB_AUTHORITY.json \
//!   --out /absolute/path/QWEN80_SOURCE_TEMPLATE_AB_AUTHORITY_REVALIDATED.json
//! ```

use hawking_core::tokenizer::Tokenizer;
use serde::{Deserialize, Serialize};
use serde_json::Value;
use sha2::{Digest, Sha256};
use std::env;
use std::error::Error;
use std::fs::{self, OpenOptions};
use std::io::Write;
use std::path::{Path, PathBuf};

const SCHEMA: &str = "hawking.ascension.qwen80_source_template_ab_authority_preflight.v1";
const STATUS: &str = "PREPARED_NOT_EXECUTED_QWEN80_SOURCE_TEMPLATE_AB_AUTHORITY";
const MODEL_ID: &str = "Qwen3-Coder-Next-80B";
const MODEL_KEY: &str = "qwen80";
const SOURCE_REPOSITORY: &str = "Qwen/Qwen3-Coder-Next";
const SOURCE_REVISION: &str = "a7fbcb5c0e12d62a448eaa0e260346bf5dcc0feb";

const CONFIG_SHA256: &str = "a7b8098d3b05777f12bb5677a26bf1240a1bb09def1b06b29e6be86cae2e84f8";
const TOKENIZER_SHA256: &str = "19564a48c4f71a2a1b937cce34c737a1e662b171c5f5d7edf641a15cd896f07d";
const TOKENIZER_CONFIG_SHA256: &str =
    "fc76878832c668e3f0f8be66e6239a475b9093d2fe5cef97c242369779e6c6e6";
const CHAT_TEMPLATE_SHA256: &str =
    "c79a833039a43602150cce0902403d6e376c50930c1b2a139b2964e1f0c322a0";
const GENERATION_CONFIG_SHA256: &str =
    "37a3c1ef63516ca489c575f0db1c0405ddc0c3dbaca9ed987344c966c37aeef5";

const TOKENIZER_VOCAB: usize = 151_669;
const LM_HEAD_VOCAB: usize = 151_936;
const RESERVED_TAIL_ROWS: usize = LM_HEAD_VOCAB - TOKENIZER_VOCAB;
const BOS_ID: u32 = 151_643;
const IM_START_ID: u32 = 151_644;
const IM_END_ID: u32 = 151_645;
const MAX_POSITION_EMBEDDINGS: usize = 262_144;
const SOURCE_TOKENIZER_MODEL_MAX_LENGTH: usize = 1_048_576;
const FUTURE_COMPLETION_TOKEN_LIMIT: usize = 16;
const MAX_PROMPT_TOKEN_LIMIT: usize = MAX_POSITION_EMBEDDINGS - FUTURE_COMPLETION_TOKEN_LIMIT;

const PROMPT_A_CONTENT: &str = "Reply with the single word copper.";
const PROMPT_B_CONTENT: &str =
    "Write a one-line Python function named add that returns the sum of a and b.";

#[derive(Clone, Debug, Deserialize, PartialEq, Eq, Serialize)]
struct SourceAuthority {
    source_dir: String,
    model_id: String,
    model_key: String,
    source_repository: String,
    source_revision: String,
    source_config_sha256: String,
    source_tokenizer_sha256: String,
    source_tokenizer_config_sha256: String,
    source_chat_template_sha256: String,
    source_generation_config_sha256: String,
    architecture: String,
    model_type: String,
    tokenizer_addressable_vocab_size: usize,
    lm_head_vocab_size: usize,
    reserved_lm_head_tail_rows: usize,
    max_position_embeddings: usize,
    source_tokenizer_model_max_length: usize,
    bos_token_id: u32,
    im_start_token_id: u32,
    im_end_token_id: u32,
    tokenizer_config_template_byte_identical_to_template_file: bool,
    tokenizer_config_template_equivalent_after_exactly_one_terminal_lf_normalization: bool,
    bounded_one_user_no_tools_source_branch_verified: bool,
}

impl SourceAuthority {
    fn expected_without_dir() -> Self {
        Self {
            source_dir: String::new(),
            model_id: MODEL_ID.into(),
            model_key: MODEL_KEY.into(),
            source_repository: SOURCE_REPOSITORY.into(),
            source_revision: SOURCE_REVISION.into(),
            source_config_sha256: CONFIG_SHA256.into(),
            source_tokenizer_sha256: TOKENIZER_SHA256.into(),
            source_tokenizer_config_sha256: TOKENIZER_CONFIG_SHA256.into(),
            source_chat_template_sha256: CHAT_TEMPLATE_SHA256.into(),
            source_generation_config_sha256: GENERATION_CONFIG_SHA256.into(),
            architecture: "Qwen3NextForCausalLM".into(),
            model_type: "qwen3_next".into(),
            tokenizer_addressable_vocab_size: TOKENIZER_VOCAB,
            lm_head_vocab_size: LM_HEAD_VOCAB,
            reserved_lm_head_tail_rows: RESERVED_TAIL_ROWS,
            max_position_embeddings: MAX_POSITION_EMBEDDINGS,
            source_tokenizer_model_max_length: SOURCE_TOKENIZER_MODEL_MAX_LENGTH,
            bos_token_id: BOS_ID,
            im_start_token_id: IM_START_ID,
            im_end_token_id: IM_END_ID,
            tokenizer_config_template_byte_identical_to_template_file: false,
            tokenizer_config_template_equivalent_after_exactly_one_terminal_lf_normalization: true,
            bounded_one_user_no_tools_source_branch_verified: true,
        }
    }

    fn exact_immutable_binding_matches(&self, other: &Self) -> bool {
        let mut left = self.clone();
        let mut right = other.clone();
        left.source_dir.clear();
        right.source_dir.clear();
        left == right
    }

    fn validate_admitted(&self) -> Result<(), String> {
        let expected = Self::expected_without_dir();
        if !self.exact_immutable_binding_matches(&expected) {
            return Err(
                "source authority is not the exact admitted Qwen3-Coder-Next tokenizer/config/chat-template binding"
                    .into(),
            );
        }
        if self.source_dir.is_empty() {
            return Err("source authority has no canonical source directory".into());
        }
        Ok(())
    }
}

#[derive(Clone, Debug, Deserialize, PartialEq, Eq, Serialize)]
struct PromptContract {
    label: String,
    source_template_mode: String,
    message_roles: Vec<String>,
    tools_present: bool,
    add_generation_prompt: bool,
    user_content: String,
    user_content_sha256: String,
    rendered_source_template_prompt: String,
    rendered_source_template_prompt_sha256: String,
    token_ids_sha256_encoding: String,
    token_ids_sha256: String,
    token_count: usize,
    token_ids: Vec<u32>,
    first_token_is_source_im_start: bool,
    source_im_start_count: usize,
    source_im_end_count: usize,
    all_token_ids_are_source_tokenizer_addressable: bool,
    exact_prompt_token_limit: usize,
    exact_future_completion_token_limit: usize,
    exact_total_context_limit: usize,
    no_truncation_or_implicit_limit_change_allowed: bool,
}

#[derive(Clone, Debug, Serialize)]
struct Revalidation {
    mode: &'static str,
    prior_contract_path: Option<String>,
    prior_contract_document_sha256: Option<String>,
    source_authority_matches_prior_exactly: bool,
    prompt_contracts_match_prior_exactly: bool,
}

#[derive(Clone, Debug, Serialize)]
struct RejectionChecks {
    reserved_lm_head_tail_id_rejected: bool,
    collapsed_ab_prompt_content_rejected: bool,
    collapsed_ab_rendered_prompt_rejected: bool,
    collapsed_ab_token_id_path_rejected: bool,
    unknown_source_revision_rejected: bool,
    unknown_chat_template_digest_rejected: bool,
    non_source_template_authority_rejected: bool,
    unsupported_role_rejected: bool,
    system_message_rejected: bool,
    tool_use_rejected: bool,
    embedded_chat_delimiter_rejected: bool,
    prompt_limit_overflow_rejected: bool,
}

#[derive(Clone, Debug, Serialize)]
struct FutureReceiptInputs {
    future_receipt_schema: &'static str,
    exact_prompt_contract_labels: Vec<&'static str>,
    current_runtime_binding_fields_required: Vec<&'static str>,
    prompt_binding_fields_required: Vec<&'static str>,
    future_execution_observations_required: Vec<&'static str>,
    future_failure_conditions: Vec<&'static str>,
    claim_boundary: Vec<&'static str>,
}

#[derive(Serialize)]
struct Report {
    schema: &'static str,
    status: &'static str,
    prepared: bool,
    executed: bool,
    source_authority: SourceAuthority,
    prompt_contracts: Vec<PromptContract>,
    contracts_are_distinct_by_content_render_and_token_ids: bool,
    revalidation: Revalidation,
    rejection_checks: RejectionChecks,
    future_receipt_inputs: FutureReceiptInputs,
    execution_boundary: ExecutionBoundary,
    unsealed_preimage_sha256: String,
}

#[derive(Serialize)]
struct ExecutionBoundary {
    only_named_small_source_sidecars_read: bool,
    live_packed_artifact_scan_performed: bool,
    raw_weight_or_gravity_payload_opened: bool,
    metal_device_or_dispatch_performed: bool,
    model_or_decoder_execution_performed: bool,
    server_or_hcli_execution_performed: bool,
    generation_or_coherence_evaluation_performed: bool,
    tps_or_tg_measurement_performed: bool,
}

#[derive(Deserialize)]
struct PriorReport {
    schema: String,
    status: String,
    source_authority: SourceAuthority,
    prompt_contracts: Vec<PromptContract>,
}

struct Arguments {
    source_dir: PathBuf,
    out: PathBuf,
    revalidate: Option<PathBuf>,
}

fn sha256_hex(bytes: &[u8]) -> String {
    format!("{:x}", Sha256::digest(bytes))
}

fn token_ids_sha256(ids: &[u32]) -> String {
    let bytes: Vec<u8> = ids.iter().flat_map(|id| id.to_le_bytes()).collect();
    sha256_hex(&bytes)
}

fn expected_file(
    path: &Path,
    expected_sha: &str,
    max_bytes: u64,
    label: &str,
) -> Result<Vec<u8>, String> {
    let metadata = fs::symlink_metadata(path)
        .map_err(|error| format!("{label} metadata failed at {}: {error}", path.display()))?;
    if !metadata.file_type().is_file() || metadata.file_type().is_symlink() {
        return Err(format!("{label} must be a regular non-symlink file"));
    }
    if metadata.len() > max_bytes {
        return Err(format!(
            "{label} is too large for a source-sidecar preflight"
        ));
    }
    let bytes = fs::read(path).map_err(|error| format!("{label} read failed: {error}"))?;
    let observed = sha256_hex(&bytes);
    if observed != expected_sha {
        return Err(format!(
            "{label} SHA-256 mismatch: expected {expected_sha}, observed {observed}"
        ));
    }
    Ok(bytes)
}

fn json_object(bytes: &[u8], label: &str) -> Result<serde_json::Map<String, Value>, String> {
    serde_json::from_slice::<Value>(bytes)
        .map_err(|error| format!("{label} is invalid JSON: {error}"))?
        .as_object()
        .cloned()
        .ok_or_else(|| format!("{label} root must be an object"))
}

fn required_str<'a>(
    object: &'a serde_json::Map<String, Value>,
    field: &str,
    label: &str,
) -> Result<&'a str, String> {
    object
        .get(field)
        .and_then(Value::as_str)
        .ok_or_else(|| format!("{label} lacks string field {field:?}"))
}

fn required_usize(
    object: &serde_json::Map<String, Value>,
    field: &str,
    label: &str,
) -> Result<usize, String> {
    object
        .get(field)
        .and_then(Value::as_u64)
        .and_then(|value| usize::try_from(value).ok())
        .ok_or_else(|| format!("{label} lacks usize field {field:?}"))
}

fn required_u32(
    object: &serde_json::Map<String, Value>,
    field: &str,
    label: &str,
) -> Result<u32, String> {
    object
        .get(field)
        .and_then(Value::as_u64)
        .and_then(|value| u32::try_from(value).ok())
        .ok_or_else(|| format!("{label} lacks u32 field {field:?}"))
}

fn added_token_matches(
    tokenizer_config: &serde_json::Map<String, Value>,
    id: u32,
    text: &str,
) -> Result<(), String> {
    let token = tokenizer_config
        .get("added_tokens_decoder")
        .and_then(Value::as_object)
        .and_then(|entries| entries.get(&id.to_string()))
        .and_then(Value::as_object)
        .ok_or_else(|| format!("source tokenizer config lacks added token {id}"))?;
    if token.get("content").and_then(Value::as_str) != Some(text)
        || token.get("special").and_then(Value::as_bool) != Some(true)
    {
        return Err(format!("source tokenizer config added token {id} drifted"));
    }
    Ok(())
}

fn source_one_user_branch_anchors(template: &str) -> bool {
    [
        "{%- for message in loop_messages %}",
        "{%- elif message.role == \"user\" or message.role == \"system\" or message.role == \"assistant\" %}",
        "{{- '<|im_start|>' + message.role + '\\n' + message.content + '<|im_end|>' + '\\n' }}",
        "{%- if add_generation_prompt %}",
        "{{- '<|im_start|>assistant\\n' }}",
    ]
    .iter()
    .all(|needle| template.contains(needle))
}

fn validate_template_sidecar_pair(embedded: &str, file: &str) -> Result<(bool, bool), String> {
    let byte_identical = embedded == file;
    let one_terminal_lf_equivalent =
        !byte_identical && !embedded.ends_with('\n') && file.strip_suffix('\n') == Some(embedded);
    if !byte_identical && !one_terminal_lf_equivalent {
        return Err(
            "tokenizer_config chat template is neither byte-identical nor exactly one terminal LF equivalent to chat_template.jinja"
                .into(),
        );
    }
    Ok((byte_identical, one_terminal_lf_equivalent))
}

fn load_source_authority(source_dir: &Path) -> Result<(SourceAuthority, Tokenizer), String> {
    if !source_dir.is_absolute() {
        return Err("--source-dir must be absolute".into());
    }
    let canonical = source_dir
        .canonicalize()
        .map_err(|error| format!("source directory canonicalization failed: {error}"))?;
    if !canonical.is_dir() {
        return Err("--source-dir is not a directory".into());
    }

    // These bounded file limits make it structurally impossible for this
    // preflight to read a weight shard under one of the admitted sidecar names.
    let config_bytes = expected_file(
        &canonical.join("config.json"),
        CONFIG_SHA256,
        64 * 1024,
        "source config",
    )?;
    let tokenizer_bytes = expected_file(
        &canonical.join("tokenizer.json"),
        TOKENIZER_SHA256,
        16 * 1024 * 1024,
        "source tokenizer",
    )?;
    let tokenizer_config_bytes = expected_file(
        &canonical.join("tokenizer_config.json"),
        TOKENIZER_CONFIG_SHA256,
        256 * 1024,
        "source tokenizer config",
    )?;
    let template_bytes = expected_file(
        &canonical.join("chat_template.jinja"),
        CHAT_TEMPLATE_SHA256,
        256 * 1024,
        "source chat template",
    )?;
    let generation_bytes = expected_file(
        &canonical.join("generation_config.json"),
        GENERATION_CONFIG_SHA256,
        64 * 1024,
        "source generation config",
    )?;

    let config = json_object(&config_bytes, "source config")?;
    let architectures = config
        .get("architectures")
        .and_then(Value::as_array)
        .ok_or("source config lacks architectures")?;
    if !architectures
        .iter()
        .any(|value| value.as_str() == Some("Qwen3NextForCausalLM"))
        || required_str(&config, "model_type", "source config")? != "qwen3_next"
        || required_usize(&config, "vocab_size", "source config")? != LM_HEAD_VOCAB
        || required_u32(&config, "bos_token_id", "source config")? != BOS_ID
        || required_u32(&config, "eos_token_id", "source config")? != IM_END_ID
        || required_usize(&config, "max_position_embeddings", "source config")?
            != MAX_POSITION_EMBEDDINGS
    {
        return Err("source config drifted from the admitted Qwen80 authority".into());
    }

    let tokenizer_config = json_object(&tokenizer_config_bytes, "source tokenizer config")?;
    if tokenizer_config
        .get("add_bos_token")
        .and_then(Value::as_bool)
        != Some(false)
        || tokenizer_config.get("eos_token").and_then(Value::as_str) != Some("<|im_end|>")
        || tokenizer_config.get("pad_token").and_then(Value::as_str) != Some("<|endoftext|>")
        || required_usize(
            &tokenizer_config,
            "model_max_length",
            "source tokenizer config",
        )? != SOURCE_TOKENIZER_MODEL_MAX_LENGTH
    {
        return Err("source tokenizer config policy drifted".into());
    }
    added_token_matches(&tokenizer_config, BOS_ID, "<|endoftext|>")?;
    added_token_matches(&tokenizer_config, IM_START_ID, "<|im_start|>")?;
    added_token_matches(&tokenizer_config, IM_END_ID, "<|im_end|>")?;

    let template = std::str::from_utf8(&template_bytes)
        .map_err(|error| format!("source chat template is not UTF-8: {error}"))?;
    let embedded = required_str(
        &tokenizer_config,
        "chat_template",
        "source tokenizer config",
    )?;
    let (byte_identical, one_terminal_lf_equivalent) =
        validate_template_sidecar_pair(embedded, template)?;
    if !source_one_user_branch_anchors(template) {
        return Err(
            "source chat template lacks the exact bounded source-user branch anchors".into(),
        );
    }

    let generation = json_object(&generation_bytes, "source generation config")?;
    if generation.get("do_sample").and_then(Value::as_bool) != Some(true)
        || generation.get("top_k").and_then(Value::as_u64) != Some(40)
        || generation.get("top_p").and_then(Value::as_f64) != Some(0.95)
        || generation.get("temperature").and_then(Value::as_f64) != Some(1.0)
    {
        return Err("source generation configuration drifted".into());
    }

    let tokenizer = Tokenizer::from_file(canonical.join("tokenizer.json"))
        .map_err(|error| format!("source tokenizer load failed: {error}"))?;
    if tokenizer.vocab_size() != TOKENIZER_VOCAB {
        return Err(format!(
            "source tokenizer addressable vocabulary {} differs from {TOKENIZER_VOCAB}",
            tokenizer.vocab_size()
        ));
    }
    for (text, id) in [
        ("<|endoftext|>", BOS_ID),
        ("<|im_start|>", IM_START_ID),
        ("<|im_end|>", IM_END_ID),
    ] {
        let ids = tokenizer
            .encode(text, false)
            .map_err(|error| format!("source special token encode failed for {text:?}: {error}"))?;
        if ids != [id] {
            return Err(format!(
                "source tokenizer special token binding drifted for {text:?}"
            ));
        }
    }

    let authority = SourceAuthority {
        source_dir: canonical.display().to_string(),
        model_id: MODEL_ID.into(),
        model_key: MODEL_KEY.into(),
        source_repository: SOURCE_REPOSITORY.into(),
        source_revision: SOURCE_REVISION.into(),
        source_config_sha256: sha256_hex(&config_bytes),
        source_tokenizer_sha256: sha256_hex(&tokenizer_bytes),
        source_tokenizer_config_sha256: sha256_hex(&tokenizer_config_bytes),
        source_chat_template_sha256: sha256_hex(&template_bytes),
        source_generation_config_sha256: sha256_hex(&generation_bytes),
        architecture: "Qwen3NextForCausalLM".into(),
        model_type: "qwen3_next".into(),
        tokenizer_addressable_vocab_size: TOKENIZER_VOCAB,
        lm_head_vocab_size: LM_HEAD_VOCAB,
        reserved_lm_head_tail_rows: RESERVED_TAIL_ROWS,
        max_position_embeddings: MAX_POSITION_EMBEDDINGS,
        source_tokenizer_model_max_length: SOURCE_TOKENIZER_MODEL_MAX_LENGTH,
        bos_token_id: BOS_ID,
        im_start_token_id: IM_START_ID,
        im_end_token_id: IM_END_ID,
        tokenizer_config_template_byte_identical_to_template_file: byte_identical,
        tokenizer_config_template_equivalent_after_exactly_one_terminal_lf_normalization:
            one_terminal_lf_equivalent,
        bounded_one_user_no_tools_source_branch_verified: true,
    };
    authority.validate_admitted()?;
    Ok((authority, tokenizer))
}

fn render_exact_source_user_branch(
    template_authority: &SourceAuthority,
    roles: &[&str],
    user_content: &str,
    tools_present: bool,
    add_generation_prompt: bool,
) -> Result<String, String> {
    template_authority.validate_admitted()?;
    if tools_present {
        return Err("source-template A/B contract rejects tool-enabled rendering".into());
    }
    if roles != ["user"] {
        return Err("source-template A/B contract supports exactly one source user role".into());
    }
    if user_content.is_empty() {
        return Err("source-template A/B contract rejects an empty user prompt".into());
    }
    if user_content.contains("<|im_start|>") || user_content.contains("<|im_end|>") {
        return Err("source-template A/B contract rejects embedded chat delimiters".into());
    }
    let suffix = if add_generation_prompt {
        "<|im_start|>assistant\n"
    } else {
        ""
    };
    Ok(format!(
        "<|im_start|>user\n{user_content}<|im_end|>\n{suffix}"
    ))
}

fn validate_prompt_ids(ids: &[u32]) -> Result<(), String> {
    if ids.is_empty() || ids.first() != Some(&IM_START_ID) {
        return Err("source template prompt must begin with <|im_start|>".into());
    }
    if ids.iter().any(|&id| id as usize >= TOKENIZER_VOCAB) {
        return Err("reserved lm-head tail/non-token ID is invalid in a source prompt".into());
    }
    let starts = ids.iter().filter(|&&id| id == IM_START_ID).count();
    let ends = ids.iter().filter(|&&id| id == IM_END_ID).count();
    if starts != 2 || ends != 1 {
        return Err("source user template control-token structure drifted".into());
    }
    Ok(())
}

fn validate_exact_prompt_limits(
    prompt_token_count: usize,
    future_completion_token_limit: usize,
) -> Result<(), String> {
    if future_completion_token_limit != FUTURE_COMPLETION_TOKEN_LIMIT {
        return Err(format!(
            "future completion limit must remain the exact A/B contract value {FUTURE_COMPLETION_TOKEN_LIMIT}"
        ));
    }
    if prompt_token_count > MAX_PROMPT_TOKEN_LIMIT
        || prompt_token_count
            .checked_add(future_completion_token_limit)
            .ok_or("prompt/future completion token limits overflow usize")?
            > MAX_POSITION_EMBEDDINGS
    {
        return Err("source-template A/B prompt exceeds the exact future context limit".into());
    }
    Ok(())
}

fn build_prompt_contract(
    authority: &SourceAuthority,
    tokenizer: &Tokenizer,
    label: &str,
    user_content: &str,
) -> Result<PromptContract, String> {
    let rendered =
        render_exact_source_user_branch(authority, &["user"], user_content, false, true)?;
    let ids = tokenizer
        .encode(&rendered, false)
        .map_err(|error| format!("source tokenizer encode failed for prompt {label}: {error}"))?;
    validate_prompt_ids(&ids)?;
    validate_exact_prompt_limits(ids.len(), FUTURE_COMPLETION_TOKEN_LIMIT)
        .map_err(|error| format!("prompt {label}: {error}"))?;
    Ok(PromptContract {
        label: label.into(),
        source_template_mode: "exact_admitted_source_one_user_no_tools_add_generation_prompt"
            .into(),
        message_roles: vec!["user".into()],
        tools_present: false,
        add_generation_prompt: true,
        user_content: user_content.into(),
        user_content_sha256: sha256_hex(user_content.as_bytes()),
        rendered_source_template_prompt_sha256: sha256_hex(rendered.as_bytes()),
        rendered_source_template_prompt: rendered,
        token_ids_sha256_encoding: "u32-little-endian-concatenation".into(),
        token_ids_sha256: token_ids_sha256(&ids),
        token_count: ids.len(),
        token_ids: ids,
        first_token_is_source_im_start: true,
        source_im_start_count: 2,
        source_im_end_count: 1,
        all_token_ids_are_source_tokenizer_addressable: true,
        exact_prompt_token_limit: MAX_PROMPT_TOKEN_LIMIT,
        exact_future_completion_token_limit: FUTURE_COMPLETION_TOKEN_LIMIT,
        exact_total_context_limit: MAX_POSITION_EMBEDDINGS,
        no_truncation_or_implicit_limit_change_allowed: true,
    })
}

fn require_distinct_contracts(a: &PromptContract, b: &PromptContract) -> Result<(), String> {
    if a.label == b.label {
        return Err("A/B prompt contracts must have distinct labels".into());
    }
    if a.user_content == b.user_content || a.user_content_sha256 == b.user_content_sha256 {
        return Err("A/B prompt contracts collapsed to identical user content".into());
    }
    if a.rendered_source_template_prompt == b.rendered_source_template_prompt
        || a.rendered_source_template_prompt_sha256 == b.rendered_source_template_prompt_sha256
    {
        return Err(
            "A/B prompt contracts collapsed to identical rendered source template prompts".into(),
        );
    }
    if a.token_ids == b.token_ids || a.token_ids_sha256 == b.token_ids_sha256 {
        return Err(
            "A/B prompt contracts collapsed to an identical source tokenizer ID path".into(),
        );
    }
    Ok(())
}

fn rejection_checks(authority: &SourceAuthority, a: &PromptContract) -> RejectionChecks {
    let reserved_lm_head_tail_id_rejected =
        validate_prompt_ids(&[IM_START_ID, TOKENIZER_VOCAB as u32]).is_err();

    let mut content_collapsed = a.clone();
    content_collapsed.label = "B".into();
    let collapsed_ab_prompt_content_rejected =
        require_distinct_contracts(a, &content_collapsed).is_err();

    let mut rendered_collapsed = a.clone();
    rendered_collapsed.label = "B".into();
    rendered_collapsed.user_content = "different source text".into();
    rendered_collapsed.user_content_sha256 = sha256_hex(rendered_collapsed.user_content.as_bytes());
    let collapsed_ab_rendered_prompt_rejected =
        require_distinct_contracts(a, &rendered_collapsed).is_err();

    let mut token_ids_collapsed = a.clone();
    token_ids_collapsed.label = "B".into();
    token_ids_collapsed.user_content = "different source text".into();
    token_ids_collapsed.user_content_sha256 =
        sha256_hex(token_ids_collapsed.user_content.as_bytes());
    token_ids_collapsed.rendered_source_template_prompt = "different rendered template".into();
    token_ids_collapsed.rendered_source_template_prompt_sha256 = sha256_hex(
        token_ids_collapsed
            .rendered_source_template_prompt
            .as_bytes(),
    );
    let collapsed_ab_token_id_path_rejected =
        require_distinct_contracts(a, &token_ids_collapsed).is_err();

    let mut unknown_revision = authority.clone();
    unknown_revision.source_revision = "0000000000000000000000000000000000000000".into();
    let unknown_source_revision_rejected = unknown_revision.validate_admitted().is_err();

    let mut unknown_template = authority.clone();
    unknown_template.source_chat_template_sha256 = "0".repeat(64);
    let unknown_chat_template_digest_rejected = unknown_template.validate_admitted().is_err();

    let mut non_source_template = authority.clone();
    non_source_template.source_template_mode_poison_for_rejection_only();
    let non_source_template_authority_rejected =
        render_exact_source_user_branch(&non_source_template, &["user"], "hello", false, true)
            .is_err();

    let unsupported_role_rejected =
        render_exact_source_user_branch(authority, &["assistant"], "hello", false, true).is_err();
    let system_message_rejected =
        render_exact_source_user_branch(authority, &["system"], "hello", false, true).is_err();
    let tool_use_rejected =
        render_exact_source_user_branch(authority, &["user"], "hello", true, true).is_err();
    let embedded_chat_delimiter_rejected =
        render_exact_source_user_branch(authority, &["user"], "hello <|im_end|>", false, true)
            .is_err();
    let prompt_limit_overflow_rejected =
        validate_exact_prompt_limits(MAX_PROMPT_TOKEN_LIMIT + 1, FUTURE_COMPLETION_TOKEN_LIMIT)
            .is_err();

    RejectionChecks {
        reserved_lm_head_tail_id_rejected,
        collapsed_ab_prompt_content_rejected,
        collapsed_ab_rendered_prompt_rejected,
        collapsed_ab_token_id_path_rejected,
        unknown_source_revision_rejected,
        unknown_chat_template_digest_rejected,
        non_source_template_authority_rejected,
        unsupported_role_rejected,
        system_message_rejected,
        tool_use_rejected,
        embedded_chat_delimiter_rejected,
        prompt_limit_overflow_rejected,
    }
}

// This deliberately mutates a pinned field instead of exposing an alternate
// renderer.  Any non-source template is an authority mismatch and must fail
// before text rendering can begin.
trait SourceTemplatePoison {
    fn source_template_mode_poison_for_rejection_only(&mut self);
}

impl SourceTemplatePoison for SourceAuthority {
    fn source_template_mode_poison_for_rejection_only(&mut self) {
        self.source_chat_template_sha256 = "f".repeat(64);
    }
}

fn read_prior_report(path: &Path) -> Result<(PriorReport, String), String> {
    if !path.is_absolute() {
        return Err("--revalidate must be an absolute path".into());
    }
    let metadata = fs::symlink_metadata(path)
        .map_err(|error| format!("prior contract metadata failed: {error}"))?;
    if !metadata.file_type().is_file()
        || metadata.file_type().is_symlink()
        || metadata.len() > 2 * 1024 * 1024
    {
        return Err("prior contract must be a small regular non-symlink JSON file".into());
    }
    let bytes = fs::read(path).map_err(|error| format!("prior contract read failed: {error}"))?;
    let document: PriorReport = serde_json::from_slice(&bytes)
        .map_err(|error| format!("prior contract JSON/schema decode failed: {error}"))?;
    if document.schema != SCHEMA || document.status != STATUS {
        return Err("prior contract has an unknown source-template A/B schema or status".into());
    }
    document.source_authority.validate_admitted()?;
    if document.prompt_contracts.len() != 2 {
        return Err("prior contract must contain exactly prompt A and prompt B".into());
    }
    if document.prompt_contracts[0].label != "A" || document.prompt_contracts[1].label != "B" {
        return Err("prior contract prompt labels/order must be exactly A then B".into());
    }
    require_distinct_contracts(&document.prompt_contracts[0], &document.prompt_contracts[1])?;
    Ok((document, sha256_hex(&bytes)))
}

fn future_receipt_inputs() -> FutureReceiptInputs {
    FutureReceiptInputs {
        future_receipt_schema: "hawking.ascension.qwen80_source_template_ab_generation_hcli_receipt.v1",
        exact_prompt_contract_labels: vec!["A", "B"],
        current_runtime_binding_fields_required: vec![
            "model_id", "model_key", "source_repository", "source_revision",
            "manifest_seal_sha256", "admission_receipt_seal_sha256",
            "runtime_receipt_seal_sha256", "runtime_executable_sha256",
        ],
        prompt_binding_fields_required: vec![
            "source_config_sha256", "source_tokenizer_sha256",
            "source_tokenizer_config_sha256", "source_chat_template_sha256",
            "source_generation_config_sha256", "message_roles", "tools_present",
            "add_generation_prompt", "rendered_source_template_prompt_sha256",
            "token_ids_sha256", "token_count", "token_ids",
            "exact_future_completion_token_limit", "exact_total_context_limit",
        ],
        future_execution_observations_required: vec![
            "actual full decoder prefill and autoregressive completion on each exact token-ID path",
            "tail mask before each sampler decision and every feedback token below tokenizer vocabulary 151669",
            "completion bytes and bytes_sha256 retained separately for A and B",
            "prompt-to-output distinction tested on actual completion bytes without treating that test as a coherence judgement",
            "same source/runtime/template binding held across both controls",
            "HCLI transport fields only if a separately qualified HCLI adapter is actually exercised",
        ],
        future_failure_conditions: vec![
            "source sidecar hash, source revision, role, limit, rendered prompt, or token-ID hash drift",
            "reserved tail ID in prompt, sampler result, or feedback",
            "A/B prompt or token-path collapse",
            "non-source template, implicit truncation, system/tool injection, fallback, synthetic input, or shadow model",
            "missing complete-token telemetry or unbound runtime identity",
        ],
        claim_boundary: vec![
            "This preflight supplies future receipt inputs only; it does not execute a model token or generate completion bytes.",
            "It cannot establish coherence, HCLI qualification, BASE_TRUE_TPS, TG, capability, or tournament eligibility.",
        ],
    }
}

fn write_new_report(path: &Path, report: &Report) -> Result<(), Box<dyn Error>> {
    if !path.is_absolute() {
        return Err("--out must be an absolute path".into());
    }
    let parent = path.parent().ok_or("output path has no parent")?;
    if !parent.is_dir() {
        return Err(format!("output parent is missing: {}", parent.display()).into());
    }
    let bytes = serde_json::to_vec_pretty(report)?;
    let mut file = OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(path)
        .map_err(|error| format!("refusing to overwrite --out {}: {error}", path.display()))?;
    file.write_all(&bytes)?;
    file.write_all(b"\n")?;
    file.sync_all()?;
    Ok(())
}

fn parse_args() -> Result<Arguments, Box<dyn Error>> {
    let mut source_dir = None;
    let mut out = None;
    let mut revalidate = None;
    let mut values = env::args().skip(1);
    while let Some(flag) = values.next() {
        let value = values
            .next()
            .ok_or_else(|| format!("missing value after {flag}"))?;
        match flag.as_str() {
            "--source-dir" => {
                if source_dir.replace(PathBuf::from(value)).is_some() {
                    return Err("--source-dir repeated".into());
                }
            }
            "--out" => {
                if out.replace(PathBuf::from(value)).is_some() {
                    return Err("--out repeated".into());
                }
            }
            "--revalidate" => {
                if revalidate.replace(PathBuf::from(value)).is_some() {
                    return Err("--revalidate repeated".into());
                }
            }
            _ => return Err("usage: ascension_qwen80_source_template_ab_authority_preflight --source-dir ABSOLUTE_PATH --out ABSOLUTE_NEW_FILE [--revalidate ABSOLUTE_PRIOR_FILE]".into()),
        }
    }
    let source_dir = source_dir.ok_or("missing --source-dir")?;
    let out = out.ok_or("missing --out")?;
    if !source_dir.is_absolute() || !out.is_absolute() {
        return Err("--source-dir and --out must be absolute".into());
    }
    Ok(Arguments {
        source_dir,
        out,
        revalidate,
    })
}

fn run(arguments: Arguments) -> Result<(), Box<dyn Error>> {
    let (authority, tokenizer) = load_source_authority(&arguments.source_dir)?;
    let prompt_a = build_prompt_contract(&authority, &tokenizer, "A", PROMPT_A_CONTENT)?;
    let prompt_b = build_prompt_contract(&authority, &tokenizer, "B", PROMPT_B_CONTENT)?;
    require_distinct_contracts(&prompt_a, &prompt_b)?;

    let revalidation = if let Some(path) = arguments.revalidate.as_deref() {
        let (prior, document_sha256) = read_prior_report(path)?;
        let source_matches = authority.exact_immutable_binding_matches(&prior.source_authority);
        let prompts_match = prior.prompt_contracts == vec![prompt_a.clone(), prompt_b.clone()];
        if !source_matches || !prompts_match {
            return Err(
                "prior source-template A/B contract drifted and cannot be revalidated".into(),
            );
        }
        Revalidation {
            mode: "REVALIDATED_EXACT_PRIOR_CONTRACT",
            prior_contract_path: Some(path.display().to_string()),
            prior_contract_document_sha256: Some(document_sha256),
            source_authority_matches_prior_exactly: true,
            prompt_contracts_match_prior_exactly: true,
        }
    } else {
        Revalidation {
            mode: "GENERATED_FRESH_SOURCE_TEMPLATE_AB_CONTRACT",
            prior_contract_path: None,
            prior_contract_document_sha256: None,
            source_authority_matches_prior_exactly: false,
            prompt_contracts_match_prior_exactly: false,
        }
    };

    let checks = rejection_checks(&authority, &prompt_a);
    let checks_pass = [
        checks.reserved_lm_head_tail_id_rejected,
        checks.collapsed_ab_prompt_content_rejected,
        checks.collapsed_ab_rendered_prompt_rejected,
        checks.collapsed_ab_token_id_path_rejected,
        checks.unknown_source_revision_rejected,
        checks.unknown_chat_template_digest_rejected,
        checks.non_source_template_authority_rejected,
        checks.unsupported_role_rejected,
        checks.system_message_rejected,
        checks.tool_use_rejected,
        checks.embedded_chat_delimiter_rejected,
        checks.prompt_limit_overflow_rejected,
    ]
    .into_iter()
    .all(|value| value);
    if !checks_pass {
        return Err("source-template A/B rejection checks failed".into());
    }

    let mut report = Report {
        schema: SCHEMA,
        status: STATUS,
        prepared: true,
        executed: false,
        source_authority: authority,
        prompt_contracts: vec![prompt_a, prompt_b],
        contracts_are_distinct_by_content_render_and_token_ids: true,
        revalidation,
        rejection_checks: checks,
        future_receipt_inputs: future_receipt_inputs(),
        execution_boundary: ExecutionBoundary {
            only_named_small_source_sidecars_read: true,
            live_packed_artifact_scan_performed: false,
            raw_weight_or_gravity_payload_opened: false,
            metal_device_or_dispatch_performed: false,
            model_or_decoder_execution_performed: false,
            server_or_hcli_execution_performed: false,
            generation_or_coherence_evaluation_performed: false,
            tps_or_tg_measurement_performed: false,
        },
        unsealed_preimage_sha256: String::new(),
    };
    report.unsealed_preimage_sha256 = sha256_hex(&serde_json::to_vec(&report)?);
    write_new_report(&arguments.out, &report)
}

fn main() {
    match parse_args().and_then(run) {
        Ok(()) => {}
        Err(error) => {
            eprintln!("ascension_qwen80_source_template_ab_authority_preflight: {error}");
            std::process::exit(2);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn admitted_authority() -> SourceAuthority {
        let mut authority = SourceAuthority::expected_without_dir();
        authority.source_dir = "/source/qwen80".into();
        authority
    }

    fn prompt(label: &str, content: &str, rendered: &str, ids: Vec<u32>) -> PromptContract {
        PromptContract {
            label: label.into(),
            source_template_mode: "exact_admitted_source_one_user_no_tools_add_generation_prompt"
                .into(),
            message_roles: vec!["user".into()],
            tools_present: false,
            add_generation_prompt: true,
            user_content: content.into(),
            user_content_sha256: sha256_hex(content.as_bytes()),
            rendered_source_template_prompt: rendered.into(),
            rendered_source_template_prompt_sha256: sha256_hex(rendered.as_bytes()),
            token_ids_sha256_encoding: "u32-little-endian-concatenation".into(),
            token_ids_sha256: token_ids_sha256(&ids),
            token_count: ids.len(),
            token_ids: ids,
            first_token_is_source_im_start: true,
            source_im_start_count: 2,
            source_im_end_count: 1,
            all_token_ids_are_source_tokenizer_addressable: true,
            exact_prompt_token_limit: MAX_PROMPT_TOKEN_LIMIT,
            exact_future_completion_token_limit: FUTURE_COMPLETION_TOKEN_LIMIT,
            exact_total_context_limit: MAX_POSITION_EMBEDDINGS,
            no_truncation_or_implicit_limit_change_allowed: true,
        }
    }

    #[test]
    fn renderer_is_exact_for_admitted_one_user_branch() {
        let authority = admitted_authority();
        assert_eq!(
            render_exact_source_user_branch(&authority, &["user"], "hello", false, true).unwrap(),
            "<|im_start|>user\nhello<|im_end|>\n<|im_start|>assistant\n"
        );
        assert!(
            render_exact_source_user_branch(&authority, &["user"], "hello", false, false).is_ok()
        );
        assert!(
            render_exact_source_user_branch(&authority, &["system"], "hello", false, true).is_err()
        );
        assert!(
            render_exact_source_user_branch(&authority, &["user"], "hello", true, true).is_err()
        );
        assert!(render_exact_source_user_branch(
            &authority,
            &["user"],
            "<|im_start|>",
            false,
            true
        )
        .is_err());
    }

    #[test]
    fn prompt_ids_reject_tail_and_control_structure_drift() {
        assert!(validate_prompt_ids(&[IM_START_ID, 9, IM_END_ID, IM_START_ID, 10]).is_ok());
        assert!(validate_prompt_ids(&[IM_START_ID, TOKENIZER_VOCAB as u32]).is_err());
        assert!(validate_prompt_ids(&[IM_START_ID, 9, IM_END_ID]).is_err());
    }

    #[test]
    fn exact_future_limits_reject_overflow_and_limit_drift() {
        assert!(validate_exact_prompt_limits(
            MAX_PROMPT_TOKEN_LIMIT,
            FUTURE_COMPLETION_TOKEN_LIMIT
        )
        .is_ok());
        assert!(validate_exact_prompt_limits(
            MAX_PROMPT_TOKEN_LIMIT + 1,
            FUTURE_COMPLETION_TOKEN_LIMIT
        )
        .is_err());
        assert!(validate_exact_prompt_limits(1, FUTURE_COMPLETION_TOKEN_LIMIT - 1).is_err());
    }

    #[test]
    fn collapsed_ab_contracts_are_rejected_at_each_identity_boundary() {
        let a = prompt(
            "A",
            "a",
            "rendered-a",
            vec![IM_START_ID, 1, IM_END_ID, IM_START_ID],
        );
        let mut same_content = a.clone();
        same_content.label = "B".into();
        assert!(require_distinct_contracts(&a, &same_content).is_err());

        let mut same_rendered = a.clone();
        same_rendered.label = "B".into();
        same_rendered.user_content = "b".into();
        same_rendered.user_content_sha256 = sha256_hex(b"b");
        assert!(require_distinct_contracts(&a, &same_rendered).is_err());

        let mut same_ids = a.clone();
        same_ids.label = "B".into();
        same_ids.user_content = "b".into();
        same_ids.user_content_sha256 = sha256_hex(b"b");
        same_ids.rendered_source_template_prompt = "rendered-b".into();
        same_ids.rendered_source_template_prompt_sha256 = sha256_hex(b"rendered-b");
        assert!(require_distinct_contracts(&a, &same_ids).is_err());
    }

    #[test]
    fn unknown_revision_and_non_source_template_are_rejected() {
        let authority = admitted_authority();
        assert!(authority.validate_admitted().is_ok());
        let mut unknown_revision = authority.clone();
        unknown_revision.source_revision = "0".repeat(40);
        assert!(unknown_revision.validate_admitted().is_err());
        let mut non_source = authority.clone();
        non_source.source_template_mode_poison_for_rejection_only();
        assert!(
            render_exact_source_user_branch(&non_source, &["user"], "hello", false, true).is_err()
        );
    }

    #[test]
    fn source_template_pair_allows_only_identity_or_one_terminal_lf() {
        assert_eq!(
            validate_template_sidecar_pair("body", "body").unwrap(),
            (true, false)
        );
        assert_eq!(
            validate_template_sidecar_pair("body", "body\n").unwrap(),
            (false, true)
        );
        assert!(validate_template_sidecar_pair("body\n", "body\n\n").is_err());
        assert!(validate_template_sidecar_pair("body", "changed").is_err());
    }

    #[test]
    fn source_pins_and_context_budget_are_exact() {
        for digest in [
            CONFIG_SHA256,
            TOKENIZER_SHA256,
            TOKENIZER_CONFIG_SHA256,
            CHAT_TEMPLATE_SHA256,
            GENERATION_CONFIG_SHA256,
        ] {
            assert_eq!(digest.len(), 64);
            assert!(digest.bytes().all(|byte| byte.is_ascii_hexdigit()));
        }
        assert_eq!(TOKENIZER_VOCAB + RESERVED_TAIL_ROWS, LM_HEAD_VOCAB);
        assert_eq!(
            MAX_PROMPT_TOKEN_LIMIT + FUTURE_COMPLETION_TOKEN_LIMIT,
            MAX_POSITION_EMBEDDINGS
        );
    }
}

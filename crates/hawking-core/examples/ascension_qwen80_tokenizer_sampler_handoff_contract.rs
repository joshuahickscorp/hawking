//! Strict, CPU-only Qwen3-Coder-Next tokenizer/chat-template -> sampler
//! handoff contract.
//!
//! This target reads only the small source sidecars (`config.json`,
//! `tokenizer.json`, `tokenizer_config.json`, `chat_template.jinja`, and
//! `generation_config.json`) whose exact digests are pinned below. It never
//! opens a packed artifact, safetensors shard, Metal device, native runtime,
//! HCLI process, watcher, or gatekeeper.
//!
//! It supports exactly the source template's one-user/no-system/no-tools
//! branch, validates the resulting real tokenizer IDs against the source's
//! 151,669-token namespace, and exercises a selected-logit deterministic
//! sampler fixture which proves that the 267 reserved lm-head tail rows are
//! masked before feedback. It is a component contract only, not a full model
//! token, generation, HCLI, TPS, TG, or capability result.

use hawking_core::tokenizer::Tokenizer;
use serde::Serialize;
use serde_json::Value;
use sha2::{Digest, Sha256};
use std::collections::BTreeMap;
use std::env;
use std::error::Error;
use std::fs;
use std::path::{Path, PathBuf};

const SCHEMA: &str = "hawking.ascension.qwen80_tokenizer_sampler_handoff_contract.v1";
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

const LM_HEAD_VOCAB: usize = 151_936;
const TOKENIZER_VOCAB: usize = 151_669;
const RESERVED_TAIL_ROWS: usize = LM_HEAD_VOCAB - TOKENIZER_VOCAB;
const BOS_ID: u32 = 151_643;
const IM_START_ID: u32 = 151_644;
const IM_END_ID: u32 = 151_645;
const FIRST_RESERVED_ID: u32 = TOKENIZER_VOCAB as u32;
const LAST_RESERVED_ID: u32 = (LM_HEAD_VOCAB - 1) as u32;
const FIXTURE_VALID_WINNER: u32 = 151_668;

#[derive(Clone, Debug, Serialize)]
struct SourceBindingReport {
    source_dir: String,
    source_repository: &'static str,
    source_revision_from_pre_admitted_source_audit: &'static str,
    config_sha256: String,
    tokenizer_sha256: String,
    tokenizer_config_sha256: String,
    chat_template_sha256: String,
    generation_config_sha256: String,
    architecture: String,
    model_type: String,
    lm_head_vocab_size: usize,
    tokenizer_addressable_vocab_size: usize,
    reserved_lm_head_tail_rows: usize,
    bos_token_id: u32,
    im_start_token_id: u32,
    im_end_token_id: u32,
    tokenizer_config_template_byte_identical_to_template_file: bool,
    tokenizer_config_template_equivalent_after_exactly_one_terminal_lf_normalization: bool,
    bounded_one_user_branch_anchors_verified: bool,
}

#[derive(Clone, Debug)]
struct SourceBinding {
    report: SourceBindingReport,
}

#[derive(Clone, Debug)]
struct PromptHandoff {
    rendered: String,
    token_ids: Vec<u32>,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum BoundedMessageRole {
    User,
    System,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum SamplerStage {
    Fresh,
    PromptValidated,
    FixtureLogitsStaged,
    ReservedTailMasked,
    Sampled,
}

struct DeterministicSamplerHandoff {
    tokenizer_vocab_size: usize,
    lm_head_vocab_size: usize,
    stage: SamplerStage,
    logits: BTreeMap<u32, f32>,
    sampled: Option<u32>,
}

#[derive(Serialize)]
struct PromptReport {
    bounded_mode: &'static str,
    rendered_prompt_sha256: String,
    rendered_prompt: String,
    token_count: usize,
    token_ids: Vec<u32>,
    first_token_is_im_start: bool,
    im_start_count: usize,
    im_end_count: usize,
    all_token_ids_are_tokenizer_addressable: bool,
    deterministic_repeat_matches: bool,
}

#[derive(Serialize)]
struct SamplerReport {
    raw_fixture_argmax_before_tail_mask: u32,
    expected_reserved_raw_argmax: u32,
    reserved_tail_mask_cutoff: u32,
    masked_fixture_argmax: u32,
    sampled_feedback_token_id: u32,
    all_selected_reserved_fixture_logits_are_negative_infinity: bool,
    sampled_id_is_tokenizer_addressable: bool,
}

#[derive(Serialize)]
struct RejectionReport {
    source_config_hash_mismatch_rejected: bool,
    source_template_mismatch_rejected: bool,
    unsupported_system_message_rejected: bool,
    unsupported_tools_rejected: bool,
    chat_delimiter_injection_rejected: bool,
    reserved_prompt_token_rejected: bool,
    sample_before_tail_mask_rejected: bool,
    wrong_tail_mask_cutoff_rejected: bool,
    reserved_tail_feedback_rejected: bool,
}

#[derive(Serialize)]
struct IntegrationContract {
    rawls_hybrid_scheduler_handoff: Vec<&'static str>,
    source_binding_requirements: Vec<&'static str>,
    sampler_requirements: Vec<&'static str>,
    claim_boundary: Vec<&'static str>,
}

#[derive(Serialize)]
struct Report {
    schema: &'static str,
    status: &'static str,
    component_only: bool,
    live_packed_artifact_scan_performed: bool,
    metal_device_or_dispatch_performed: bool,
    source_binding: SourceBindingReport,
    prompt: PromptReport,
    sampler_fixture: SamplerReport,
    rejection_tests: RejectionReport,
    integration_contract: IntegrationContract,
    unsealed_preimage_sha256: String,
}

fn sha256_hex(bytes: &[u8]) -> String {
    format!("{:x}", Sha256::digest(bytes))
}

fn verify_expected_digest(bytes: &[u8], expected_sha: &str, label: &str) -> Result<(), String> {
    let observed = sha256_hex(bytes);
    if observed != expected_sha {
        return Err(format!(
            "{label} SHA-256 mismatch: expected {expected_sha}, observed {observed}"
        ));
    }
    Ok(())
}

fn expected_file(path: &Path, expected_sha: &str, label: &str) -> Result<Vec<u8>, String> {
    let metadata = fs::symlink_metadata(path)
        .map_err(|error| format!("{label} metadata failed at {}: {error}", path.display()))?;
    if !metadata.file_type().is_file() || metadata.file_type().is_symlink() {
        return Err(format!("{label} must be a regular non-symlink file"));
    }
    let bytes = fs::read(path).map_err(|error| format!("{label} read failed: {error}"))?;
    verify_expected_digest(&bytes, expected_sha, label)?;
    Ok(bytes)
}

fn json_object(bytes: &[u8], label: &str) -> Result<serde_json::Map<String, Value>, String> {
    serde_json::from_slice::<Value>(bytes)
        .map_err(|error| format!("{label} is invalid JSON: {error}"))?
        .as_object()
        .cloned()
        .ok_or_else(|| format!("{label} root must be an object"))
}

fn required_string<'a>(
    object: &'a serde_json::Map<String, Value>,
    field: &str,
    label: &str,
) -> Result<&'a str, String> {
    object
        .get(field)
        .and_then(Value::as_str)
        .ok_or_else(|| format!("{label} missing string {field:?}"))
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
        .ok_or_else(|| format!("{label} missing u32 {field:?}"))
}

fn added_token_matches(
    tokenizer_config: &serde_json::Map<String, Value>,
    id: u32,
    text: &str,
) -> Result<(), String> {
    let entry = tokenizer_config
        .get("added_tokens_decoder")
        .and_then(Value::as_object)
        .and_then(|tokens| tokens.get(&id.to_string()))
        .and_then(Value::as_object)
        .ok_or_else(|| format!("tokenizer_config lacks added token {id}"))?;
    if entry.get("content").and_then(Value::as_str) != Some(text)
        || entry.get("special").and_then(Value::as_bool) != Some(true)
    {
        return Err(format!(
            "tokenizer_config added token {id} drifted from {text:?}"
        ));
    }
    Ok(())
}

fn contains_required_one_user_anchors(template: &str) -> bool {
    [
        "{%- for message in loop_messages %}",
        "{{- '<|im_start|>' + message.role + '\\n' + message.content + '<|im_end|>' + '\\n' }}",
        "{%- if add_generation_prompt %}",
        "{{- '<|im_start|>assistant\\n' }}",
    ]
    .iter()
    .all(|needle| template.contains(needle))
}

fn validate_template_sidecar_pair(
    embedded_template: &str,
    template_file: &str,
) -> Result<(bool, bool), String> {
    let byte_identical = embedded_template == template_file;
    // The independently pinned source sidecars currently differ only in the
    // `chat_template.jinja` file's one terminal LF: the tokenizer config
    // embeds 6067 bytes and the companion Jinja file has those exact bytes
    // plus `\n`. Treat no other normalization as legal.
    let exactly_one_terminal_lf_equivalent = !byte_identical
        && !embedded_template.ends_with('\n')
        && template_file.strip_suffix('\n') == Some(embedded_template);
    if !byte_identical && !exactly_one_terminal_lf_equivalent {
        return Err(format!(
            "tokenizer_config chat_template is neither byte-identical nor exactly-one-terminal-LF equivalent to chat_template.jinja (embedded={} bytes sha256={}, file={} bytes sha256={})",
            embedded_template.len(),
            sha256_hex(embedded_template.as_bytes()),
            template_file.len(),
            sha256_hex(template_file.as_bytes()),
        ));
    }
    Ok((byte_identical, exactly_one_terminal_lf_equivalent))
}

fn load_source_binding(source_dir: &Path) -> Result<(SourceBinding, Tokenizer), String> {
    if !source_dir.is_absolute() {
        return Err("--source-dir must be absolute".into());
    }
    let source_dir = source_dir
        .canonicalize()
        .map_err(|error| format!("source directory canonicalization failed: {error}"))?;
    if !source_dir.is_dir() {
        return Err("--source-dir is not a directory".into());
    }
    let config_bytes = expected_file(
        &source_dir.join("config.json"),
        CONFIG_SHA256,
        "source config",
    )?;
    let tokenizer_bytes = expected_file(
        &source_dir.join("tokenizer.json"),
        TOKENIZER_SHA256,
        "source tokenizer",
    )?;
    let tokenizer_config_bytes = expected_file(
        &source_dir.join("tokenizer_config.json"),
        TOKENIZER_CONFIG_SHA256,
        "source tokenizer config",
    )?;
    let template_bytes = expected_file(
        &source_dir.join("chat_template.jinja"),
        CHAT_TEMPLATE_SHA256,
        "source chat template",
    )?;
    let generation_bytes = expected_file(
        &source_dir.join("generation_config.json"),
        GENERATION_CONFIG_SHA256,
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
        || required_string(&config, "model_type", "source config")? != "qwen3_next"
        || required_u32(&config, "vocab_size", "source config")? as usize != LM_HEAD_VOCAB
        || required_u32(&config, "bos_token_id", "source config")? != BOS_ID
        || required_u32(&config, "eos_token_id", "source config")? != IM_END_ID
    {
        return Err(
            "source config does not match exact Qwen3-Coder-Next tokenizer/lm-head geometry".into(),
        );
    }
    let tokenizer_config = json_object(&tokenizer_config_bytes, "source tokenizer config")?;
    let template = std::str::from_utf8(&template_bytes)
        .map_err(|error| format!("source chat template is not UTF-8: {error}"))?;
    let embedded_template = required_string(
        &tokenizer_config,
        "chat_template",
        "source tokenizer config",
    )?;
    let (template_byte_identical, template_equivalent_after_exactly_one_terminal_lf_normalization) =
        validate_template_sidecar_pair(embedded_template, template)?;
    if tokenizer_config
        .get("add_bos_token")
        .and_then(Value::as_bool)
        != Some(false)
        || tokenizer_config.get("eos_token").and_then(Value::as_str) != Some("<|im_end|>")
        || tokenizer_config.get("pad_token").and_then(Value::as_str) != Some("<|endoftext|>")
    {
        return Err("source tokenizer config BOS/EOS/PAD policy drifted".into());
    }
    added_token_matches(&tokenizer_config, BOS_ID, "<|endoftext|>")?;
    added_token_matches(&tokenizer_config, IM_START_ID, "<|im_start|>")?;
    added_token_matches(&tokenizer_config, IM_END_ID, "<|im_end|>")?;
    if !contains_required_one_user_anchors(template) {
        return Err("source chat template lacks bounded one-user/no-tools source anchors".into());
    }
    let generation = json_object(&generation_bytes, "source generation config")?;
    if generation.get("do_sample").and_then(Value::as_bool) != Some(true)
        || generation.get("top_k").and_then(Value::as_u64) != Some(40)
        || generation.get("top_p").and_then(Value::as_f64) != Some(0.95)
        || generation.get("temperature").and_then(Value::as_f64) != Some(1.0)
    {
        return Err(
            "source generation configuration drifted from current Qwen80 source facts".into(),
        );
    }
    let tokenizer = Tokenizer::from_file(source_dir.join("tokenizer.json"))
        .map_err(|error| format!("source tokenizer load failed: {error}"))?;
    if tokenizer.vocab_size() != TOKENIZER_VOCAB {
        return Err(format!(
            "source tokenizer addressable vocabulary {} differs from expected {TOKENIZER_VOCAB}",
            tokenizer.vocab_size()
        ));
    }
    for (text, id) in [
        ("<|endoftext|>", BOS_ID),
        ("<|im_start|>", IM_START_ID),
        ("<|im_end|>", IM_END_ID),
    ] {
        let encoded = tokenizer
            .encode(text, false)
            .map_err(|error| format!("special token encode failed for {text:?}: {error}"))?;
        if encoded != [id] {
            return Err(format!(
                "source tokenizer does not encode {text:?} as [{id}]"
            ));
        }
    }
    Ok((
        SourceBinding {
            report: SourceBindingReport {
                source_dir: source_dir.display().to_string(),
                source_repository: SOURCE_REPOSITORY,
                source_revision_from_pre_admitted_source_audit: SOURCE_REVISION,
                config_sha256: sha256_hex(&config_bytes),
                tokenizer_sha256: sha256_hex(&tokenizer_bytes),
                tokenizer_config_sha256: sha256_hex(&tokenizer_config_bytes),
                chat_template_sha256: sha256_hex(&template_bytes),
                generation_config_sha256: sha256_hex(&generation_bytes),
                architecture: "Qwen3NextForCausalLM".to_owned(),
                model_type: "qwen3_next".to_owned(),
                lm_head_vocab_size: LM_HEAD_VOCAB,
                tokenizer_addressable_vocab_size: TOKENIZER_VOCAB,
                reserved_lm_head_tail_rows: RESERVED_TAIL_ROWS,
                bos_token_id: BOS_ID,
                im_start_token_id: IM_START_ID,
                im_end_token_id: IM_END_ID,
                tokenizer_config_template_byte_identical_to_template_file: template_byte_identical,
                tokenizer_config_template_equivalent_after_exactly_one_terminal_lf_normalization:
                    template_equivalent_after_exactly_one_terminal_lf_normalization,
                bounded_one_user_branch_anchors_verified: true,
            },
        },
        tokenizer,
    ))
}

fn render_bounded_messages(
    messages: &[(BoundedMessageRole, &str)],
    tools_present: bool,
) -> Result<String, String> {
    if tools_present {
        return Err("bounded source-template handoff rejects tools".into());
    }
    let [(role, user_content)] = messages else {
        return Err("bounded source-template handoff supports exactly one user message".into());
    };
    if *role != BoundedMessageRole::User {
        return Err("bounded source-template handoff rejects non-user message roles".into());
    }
    if user_content.is_empty() {
        return Err("bounded source-template handoff rejects empty user content".into());
    }
    // The real source branch does not claim to sanitize role delimiters. This
    // HCLI boundary refuses them rather than silently presenting injected
    // control markup as an ordinary user message.
    if user_content.contains("<|im_start|>") || user_content.contains("<|im_end|>") {
        return Err("bounded source-template handoff rejects embedded chat delimiters".into());
    }
    Ok(format!(
        "<|im_start|>user\n{user_content}<|im_end|>\n<|im_start|>assistant\n"
    ))
}

fn render_bounded_one_user_no_tools(user_content: &str) -> Result<String, String> {
    render_bounded_messages(&[(BoundedMessageRole::User, user_content)], false)
}

fn validate_prompt_token_ids(ids: &[u32]) -> Result<(), String> {
    if ids.is_empty() || ids.first() != Some(&IM_START_ID) {
        return Err("bounded chat prompt must begin with source <|im_start|>".into());
    }
    if ids.iter().any(|&id| id as usize >= TOKENIZER_VOCAB) {
        return Err("bounded chat prompt contains an lm-head reserved-tail/non-token id".into());
    }
    let starts = ids.iter().filter(|&&id| id == IM_START_ID).count();
    let ends = ids.iter().filter(|&&id| id == IM_END_ID).count();
    if starts != 2 || ends != 1 {
        return Err("bounded one-user template control-token structure drifted".into());
    }
    Ok(())
}

fn format_and_tokenize(tokenizer: &Tokenizer, content: &str) -> Result<PromptHandoff, String> {
    let rendered = render_bounded_one_user_no_tools(content)?;
    let token_ids = tokenizer
        .encode(&rendered, false)
        .map_err(|error| format!("source tokenizer prompt encode failed: {error}"))?;
    validate_prompt_token_ids(&token_ids)?;
    Ok(PromptHandoff {
        rendered,
        token_ids,
    })
}

impl DeterministicSamplerHandoff {
    fn new() -> Self {
        Self {
            tokenizer_vocab_size: TOKENIZER_VOCAB,
            lm_head_vocab_size: LM_HEAD_VOCAB,
            stage: SamplerStage::Fresh,
            logits: BTreeMap::new(),
            sampled: None,
        }
    }

    fn begin_prompt(&mut self, token_ids: &[u32]) -> Result<(), String> {
        if self.stage != SamplerStage::Fresh {
            return Err(
                "sampler handoff prompt may be accepted only once at the fresh stage".into(),
            );
        }
        validate_prompt_token_ids(token_ids)?;
        self.stage = SamplerStage::PromptValidated;
        Ok(())
    }

    fn stage_selected_logit_fixture(&mut self) -> Result<(), String> {
        if self.stage != SamplerStage::PromptValidated {
            return Err("selected-logit fixture requires validated prompt tokens".into());
        }
        // Tail values intentionally win before masking. They represent only a
        // deterministic contract fixture, never a partial lm-head result.
        self.logits = BTreeMap::from([
            (17, 0.5),
            (FIXTURE_VALID_WINNER, 8.0),
            (FIRST_RESERVED_ID, 9.0),
            (LAST_RESERVED_ID, 99.0),
        ]);
        self.stage = SamplerStage::FixtureLogitsStaged;
        Ok(())
    }

    fn raw_fixture_argmax(&self) -> Result<u32, String> {
        if self.stage != SamplerStage::FixtureLogitsStaged
            && self.stage != SamplerStage::ReservedTailMasked
            && self.stage != SamplerStage::Sampled
        {
            return Err("raw fixture argmax requires staged logits".into());
        }
        self.argmax(false)
    }

    fn mask_reserved_lm_head_tail(&mut self, first_reserved_id: u32) -> Result<(), String> {
        if self.stage != SamplerStage::FixtureLogitsStaged {
            return Err("reserved-tail mask requires staged logits before any sampling".into());
        }
        if first_reserved_id as usize != self.tokenizer_vocab_size {
            return Err("reserved-tail mask cutoff differs from source tokenizer namespace".into());
        }
        for (&id, logit) in &mut self.logits {
            if id >= first_reserved_id {
                *logit = f32::NEG_INFINITY;
            }
        }
        self.stage = SamplerStage::ReservedTailMasked;
        Ok(())
    }

    fn argmax(&self, tokenizer_only: bool) -> Result<u32, String> {
        self.logits
            .iter()
            .filter(|(&id, &logit)| {
                logit.is_finite() && (!tokenizer_only || (id as usize) < self.tokenizer_vocab_size)
            })
            .max_by(|(left_id, left), (right_id, right)| {
                left.total_cmp(right).then_with(|| right_id.cmp(left_id))
            })
            .map(|(&id, _)| id)
            .ok_or_else(|| "sampler fixture has no finite eligible logit".into())
    }

    fn sample_feedback_token(&mut self) -> Result<u32, String> {
        if self.stage != SamplerStage::ReservedTailMasked {
            return Err("sampler must refuse output before reserved-tail masking".into());
        }
        let sampled = self.argmax(true)?;
        if sampled as usize >= self.tokenizer_vocab_size
            || sampled as usize >= self.lm_head_vocab_size
        {
            return Err("sampler selected a reserved/non-token ID".into());
        }
        self.sampled = Some(sampled);
        self.stage = SamplerStage::Sampled;
        Ok(sampled)
    }

    fn feedback_token(&self) -> Result<u32, String> {
        if self.stage != SamplerStage::Sampled {
            return Err("feedback token is unavailable before a valid masked sample".into());
        }
        let token = self.sampled.ok_or("sampled stage lacks token")?;
        if token as usize >= self.tokenizer_vocab_size {
            return Err("feedback token is in source-reserved tail".into());
        }
        Ok(token)
    }

    fn selected_tail_masked(&self) -> bool {
        self.logits.iter().all(|(&id, &logit)| {
            (id as usize) < self.tokenizer_vocab_size || logit == f32::NEG_INFINITY
        })
    }
}

fn prompt_report(tokenizer: &Tokenizer) -> Result<(PromptReport, PromptHandoff), String> {
    const CONTENT: &str = "Return the word ready.";
    let first = format_and_tokenize(tokenizer, CONTENT)?;
    let second = format_and_tokenize(tokenizer, CONTENT)?;
    let starts = first
        .token_ids
        .iter()
        .filter(|&&id| id == IM_START_ID)
        .count();
    let ends = first
        .token_ids
        .iter()
        .filter(|&&id| id == IM_END_ID)
        .count();
    let report = PromptReport {
        bounded_mode: "exact_source_one_user_no_system_no_tools_branch_plus_hcli_delimiter_refusal",
        rendered_prompt_sha256: sha256_hex(first.rendered.as_bytes()),
        rendered_prompt: first.rendered.clone(),
        token_count: first.token_ids.len(),
        token_ids: first.token_ids.clone(),
        first_token_is_im_start: first.token_ids.first() == Some(&IM_START_ID),
        im_start_count: starts,
        im_end_count: ends,
        all_token_ids_are_tokenizer_addressable: first
            .token_ids
            .iter()
            .all(|&id| (id as usize) < TOKENIZER_VOCAB),
        deterministic_repeat_matches: first.rendered == second.rendered
            && first.token_ids == second.token_ids,
    };
    Ok((report, first))
}

fn sampler_report(prompt: &PromptHandoff) -> Result<SamplerReport, String> {
    let mut handoff = DeterministicSamplerHandoff::new();
    handoff.begin_prompt(&prompt.token_ids)?;
    handoff.stage_selected_logit_fixture()?;
    let raw_fixture_argmax_before_tail_mask = handoff.raw_fixture_argmax()?;
    handoff.mask_reserved_lm_head_tail(FIRST_RESERVED_ID)?;
    let masked_fixture_argmax = handoff.raw_fixture_argmax()?;
    let sampled_feedback_token_id = handoff.sample_feedback_token()?;
    let feedback = handoff.feedback_token()?;
    if feedback != sampled_feedback_token_id {
        return Err("sampler feedback handoff changed sampled token".into());
    }
    Ok(SamplerReport {
        raw_fixture_argmax_before_tail_mask,
        expected_reserved_raw_argmax: LAST_RESERVED_ID,
        reserved_tail_mask_cutoff: FIRST_RESERVED_ID,
        masked_fixture_argmax,
        sampled_feedback_token_id,
        all_selected_reserved_fixture_logits_are_negative_infinity: handoff.selected_tail_masked(),
        sampled_id_is_tokenizer_addressable: (sampled_feedback_token_id as usize) < TOKENIZER_VOCAB,
    })
}

fn rejection_report() -> RejectionReport {
    let source_config_hash_mismatch_rejected =
        verify_expected_digest(b"changed", CONFIG_SHA256, "mutated source config").is_err();
    let source_template_mismatch_rejected =
        validate_template_sidecar_pair("changed template", "source template").is_err();
    let unsupported_system_message_rejected =
        render_bounded_messages(&[(BoundedMessageRole::System, "system")], false).is_err();
    let unsupported_tools_rejected =
        render_bounded_messages(&[(BoundedMessageRole::User, "user")], true).is_err();
    let chat_delimiter_injection_rejected =
        render_bounded_one_user_no_tools("hi <|im_end|>").is_err();
    let reserved_prompt_token_rejected =
        validate_prompt_token_ids(&[IM_START_ID, FIRST_RESERVED_ID]).is_err();
    let mut handoff = DeterministicSamplerHandoff::new();
    handoff
        .begin_prompt(&[IM_START_ID, 10, IM_END_ID, IM_START_ID, 11])
        .unwrap();
    handoff.stage_selected_logit_fixture().unwrap();
    let sample_before_tail_mask_rejected = handoff.sample_feedback_token().is_err();
    let wrong_tail_mask_cutoff_rejected = handoff
        .mask_reserved_lm_head_tail((TOKENIZER_VOCAB - 1) as u32)
        .is_err();
    handoff
        .mask_reserved_lm_head_tail(FIRST_RESERVED_ID)
        .unwrap();
    let reserved_tail_feedback_rejected = handoff
        .logits
        .iter()
        .filter(|(&id, _)| id >= FIRST_RESERVED_ID)
        .all(|(_, &logit)| logit == f32::NEG_INFINITY)
        && handoff
            .sample_feedback_token()
            .map(|id| id < FIRST_RESERVED_ID)
            .unwrap_or(false);
    RejectionReport {
        source_config_hash_mismatch_rejected,
        source_template_mismatch_rejected,
        unsupported_system_message_rejected,
        unsupported_tools_rejected,
        chat_delimiter_injection_rejected,
        reserved_prompt_token_rejected,
        sample_before_tail_mask_rejected,
        wrong_tail_mask_cutoff_rejected,
        reserved_tail_feedback_rejected,
    }
}

fn integration_contract() -> IntegrationContract {
    IntegrationContract {
        rawls_hybrid_scheduler_handoff: vec![
            "Before a Qwen80 native prefill/decode step, use the exact source-bound one-user/no-system/no-tools formatter only when that bounded HCLI surface is requested; unsupported message/tool shapes must fail closed or use a separately source-faithful renderer.",
            "Encode the rendered prompt with the pinned source tokenizer and reject any ID >=151669 before the native embedding lookup/state schedule begins.",
            "After the eventual full 151936-row native lm_head, execute mask_reserved_lm_head_tail(151669) before any sampler policy, then validate sampled_id <151669 before autoregressive feedback.",
            "This contract's deterministic argmax fixture is a test seam. Production sampling must separately honor the explicitly selected HCLI/source sampling policy and earn native full-token parity.",
        ],
        source_binding_requirements: vec![
            "Bind source sidecars by all five pinned SHA-256 values and source-audit revision a7fbcb5c0e12d62a448eaa0e260346bf5dcc0feb; a sidecar/template update must reopen this contract.",
            "Require Qwen3NextForCausalLM / qwen3_next, lm_head vocab 151936, tokenizer vocab 151669, and a 267-row reserved tail; do not synthesize token text for tail IDs.",
            "Require the two separately pinned template sidecars to be byte-identical or to differ only by chat_template.jinja's exactly one terminal LF; validate the source special-token IDs 151643/151644/151645.",
        ],
        sampler_requirements: vec![
            "Mask every full logit-domain row 151669..151935, not only rows observed in this selected-logit fixture.",
            "Sampling before masking, sampling an unaddressable ID, or forwarding it to the next token must fail closed.",
            "The direct-packed final-head component remains a separate prerequisite; this tokenizer/sampler handoff does not supply logits or an lm-head implementation.",
        ],
        claim_boundary: vec![
            "No packed Qwen80 artifact or safetensors shard was opened; no Metal context, model runtime, HCLI process, benchmark, or TPS measurement occurred.",
            "This receipt is tokenizer/template/sampler-handoff evidence only and cannot qualify Qwen80 for generation, capability, HCLI, TG, Agent OS, or tournament gates.",
        ],
    }
}

fn write_report_atomic(path: &Path, report: &Report) -> Result<(), Box<dyn Error>> {
    let parent = path.parent().ok_or("output path has no parent")?;
    if !parent.is_dir() {
        return Err(format!("output parent is missing: {}", parent.display()).into());
    }
    let temporary = path.with_extension(format!("tmp-{}", std::process::id()));
    fs::write(&temporary, serde_json::to_vec_pretty(report)?)?;
    fs::rename(&temporary, path)?;
    Ok(())
}

fn parse_args() -> Result<(PathBuf, PathBuf), Box<dyn Error>> {
    let mut source_dir = None;
    let mut out = None;
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
            _ => return Err("usage: ascension_qwen80_tokenizer_sampler_handoff_contract --source-dir ABSOLUTE_PATH --out ABSOLUTE_PATH".into()),
        }
    }
    let source_dir = source_dir.ok_or("missing --source-dir")?;
    let out = out.ok_or("missing --out")?;
    if !source_dir.is_absolute() || !out.is_absolute() {
        return Err("--source-dir and --out must be absolute".into());
    }
    Ok((source_dir, out))
}

fn run(source_dir: PathBuf, out: PathBuf) -> Result<(), Box<dyn Error>> {
    let (binding, tokenizer) = load_source_binding(&source_dir)?;
    let (prompt, handoff) = prompt_report(&tokenizer)?;
    let sampler_fixture = sampler_report(&handoff)?;
    let rejection_tests = rejection_report();
    let rejections_pass = [
        rejection_tests.source_config_hash_mismatch_rejected,
        rejection_tests.source_template_mismatch_rejected,
        rejection_tests.unsupported_system_message_rejected,
        rejection_tests.unsupported_tools_rejected,
        rejection_tests.chat_delimiter_injection_rejected,
        rejection_tests.reserved_prompt_token_rejected,
        rejection_tests.sample_before_tail_mask_rejected,
        rejection_tests.wrong_tail_mask_cutoff_rejected,
        rejection_tests.reserved_tail_feedback_rejected,
    ]
    .into_iter()
    .all(|passed| passed);
    if !prompt.first_token_is_im_start
        || prompt.im_start_count != 2
        || prompt.im_end_count != 1
        || !prompt.all_token_ids_are_tokenizer_addressable
        || !prompt.deterministic_repeat_matches
        || sampler_fixture.raw_fixture_argmax_before_tail_mask != LAST_RESERVED_ID
        || sampler_fixture.masked_fixture_argmax != FIXTURE_VALID_WINNER
        || sampler_fixture.sampled_feedback_token_id != FIXTURE_VALID_WINNER
        || !sampler_fixture.all_selected_reserved_fixture_logits_are_negative_infinity
        || !sampler_fixture.sampled_id_is_tokenizer_addressable
        || !rejections_pass
    {
        return Err("Qwen80 tokenizer/sampler handoff fixture acceptance failed".into());
    }
    let mut report = Report {
        schema: SCHEMA,
        status:
            "EARNED_SOURCE_BOUND_TOKENIZER_TEMPLATE_SAMPLER_HANDOFF_COMPONENT_NOT_RUNTIME_OR_TOKEN",
        component_only: true,
        live_packed_artifact_scan_performed: false,
        metal_device_or_dispatch_performed: false,
        source_binding: binding.report,
        prompt,
        sampler_fixture,
        rejection_tests,
        integration_contract: integration_contract(),
        unsealed_preimage_sha256: String::new(),
    };
    report.unsealed_preimage_sha256 = sha256_hex(&serde_json::to_vec(&report)?);
    write_report_atomic(&out, &report)
}

fn main() {
    match parse_args().and_then(|(source_dir, out)| run(source_dir, out)) {
        Ok(()) => {}
        Err(error) => {
            eprintln!("ascension_qwen80_tokenizer_sampler_handoff_contract: {error}");
            std::process::exit(2);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn bounded_renderer_is_exact_for_one_user_branch() {
        assert_eq!(
            render_bounded_one_user_no_tools("hello").unwrap(),
            "<|im_start|>user\nhello<|im_end|>\n<|im_start|>assistant\n"
        );
        assert!(render_bounded_one_user_no_tools("").is_err());
        assert!(render_bounded_one_user_no_tools("x <|im_start|>").is_err());
        assert!(render_bounded_messages(&[(BoundedMessageRole::System, "x")], false).is_err());
        assert!(render_bounded_messages(&[(BoundedMessageRole::User, "x")], true).is_err());
        assert!(render_bounded_messages(
            &[
                (BoundedMessageRole::User, "x"),
                (BoundedMessageRole::User, "y")
            ],
            false,
        )
        .is_err());
    }

    #[test]
    fn prompt_ids_reject_reserved_tail_and_wrong_control_structure() {
        assert!(validate_prompt_token_ids(&[IM_START_ID, 9, IM_END_ID, IM_START_ID, 10]).is_ok());
        assert!(validate_prompt_token_ids(&[IM_START_ID, FIRST_RESERVED_ID]).is_err());
        assert!(validate_prompt_token_ids(&[IM_START_ID, 9, IM_END_ID]).is_err());
    }

    #[test]
    fn sampler_fixture_masks_tail_before_feedback() {
        let prompt = [IM_START_ID, 9, IM_END_ID, IM_START_ID, 10];
        let mut handoff = DeterministicSamplerHandoff::new();
        handoff.begin_prompt(&prompt).unwrap();
        handoff.stage_selected_logit_fixture().unwrap();
        assert_eq!(handoff.raw_fixture_argmax().unwrap(), LAST_RESERVED_ID);
        assert!(handoff.sample_feedback_token().is_err());
        assert!(handoff
            .mask_reserved_lm_head_tail((TOKENIZER_VOCAB - 1) as u32)
            .is_err());
        handoff
            .mask_reserved_lm_head_tail(FIRST_RESERVED_ID)
            .unwrap();
        assert_eq!(
            handoff.sample_feedback_token().unwrap(),
            FIXTURE_VALID_WINNER
        );
        assert_eq!(handoff.feedback_token().unwrap(), FIXTURE_VALID_WINNER);
        assert!(handoff.selected_tail_masked());
    }

    #[test]
    fn source_template_anchor_set_is_strict() {
        let exact_minimal = "{%- for message in loop_messages %}\n{{- '<|im_start|>' + message.role + '\\n' + message.content + '<|im_end|>' + '\\n' }}\n{%- if add_generation_prompt %}\n{{- '<|im_start|>assistant\\n' }}";
        assert!(contains_required_one_user_anchors(exact_minimal));
        assert!(!contains_required_one_user_anchors("<|im_start|>assistant"));
    }

    #[test]
    fn sidecar_template_pair_allows_only_identity_or_one_terminal_lf() {
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
    fn source_pins_are_sha_shaped_and_tail_partition_is_exact() {
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
        assert_eq!(FIRST_RESERVED_ID, 151_669);
        assert_eq!(LAST_RESERVED_ID, 151_935);
    }
}

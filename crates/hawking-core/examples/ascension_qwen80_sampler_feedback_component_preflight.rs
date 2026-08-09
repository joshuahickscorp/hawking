//! Static/CPU-only Qwen3-Coder-Next sampler + autoregressive-feedback preflight.
//!
//! This component is intentionally narrower than a terminal-head execution.
//! It consumes one already-emitted terminal-head *contract* as external
//! authority, then fixes the future ABI between an all-row, tail-masked
//! `[151936]` logit vector and one caller-owned logical-session token buffer:
//!
//! ```text
//! future all-row terminal head -> exact tail mask -> deterministic sampler
//!                              -> validated token ID -> caller feedback slot
//!                                                         -> exact rollback
//! ```
//!
//! It opens only the explicitly named JSON terminal-head contract.  It does
//! not open model artifacts or source sidecars, construct a tokenizer, launch
//! a runtime/watcher/server, touch a kernel registry, create a Metal object,
//! acquire a lease, dispatch GPU work, generate a token, invoke HCLI, or
//! measure TPS/TG.  The host-side tests use synthetic logits solely to prove
//! the ABI/state-machine rejection boundaries.  This is therefore a
//! `PREPARED/INCOMPLETE` component contract, not decoder evidence.

use serde::Serialize;
use serde_json::{Map, Value};
use sha2::{Digest, Sha256};
use std::env;
use std::error::Error;
use std::fs::{self, OpenOptions};
use std::io::Write;
use std::path::{Path, PathBuf};

const SCHEMA: &str = "hawking.ascension.qwen80_sampler_feedback_component_preflight.v1";
const STATUS: &str =
    "PREPARED_INCOMPLETE_QWEN80_TAIL_MASKED_DETERMINISTIC_SAMPLER_CALLER_OWNED_FEEDBACK_ROLLBACK_CONTRACT";
const FUTURE_INPUT_RECEIPT_SCHEMA: &str =
    "hawking.ascension.qwen80_tail_masked_logit_input_receipt.v1";
const FUTURE_OUTPUT_RECEIPT_SCHEMA: &str =
    "hawking.ascension.qwen80_sampler_feedback_component_result.v1";
const TERMINAL_HEAD_CONTRACT_SCHEMA: &str =
    "hawking.ascension.qwen80_terminal_head_device_contract.v1";
const TERMINAL_HEAD_CONTRACT_STATUS: &str =
    "CONTRACT_ONLY_CURRENT_CPU_COMPONENTS_CONSUMED_FUTURE_DEVICE_DISPATCH_BLOCKED_NOT_RUNTIME_NO_TOKEN_NO_HCLI_NO_TPS";
const MODEL_ID: &str = "Qwen3-Coder-Next-80B";
const SOURCE_REPOSITORY: &str = "Qwen/Qwen3-Coder-Next";
const SOURCE_REVISION: &str = "a7fbcb5c0e12d62a448eaa0e260346bf5dcc0feb";
const SOURCE_CONFIG_SHA256: &str =
    "a7b8098d3b05777f12bb5677a26bf1240a1bb09def1b06b29e6be86cae2e84f8";
const SOURCE_TOKENIZER_SHA256: &str =
    "19564a48c4f71a2a1b937cce34c737a1e662b171c5f5d7edf641a15cd896f07d";
const SOURCE_TOKENIZER_CONFIG_SHA256: &str =
    "fc76878832c668e3f0f8be66e6239a475b9093d2fe5cef97c242369779e6c6e6";
const SOURCE_CHAT_TEMPLATE_SHA256: &str =
    "c79a833039a43602150cce0902403d6e376c50930c1b2a139b2964e1f0c322a0";
const SOURCE_GENERATION_CONFIG_SHA256: &str =
    "37a3c1ef63516ca489c575f0db1c0405ddc0c3dbaca9ed987344c966c37aeef5";
const LM_HEAD_VOCAB: usize = 151_936;
const TOKENIZER_VOCAB: usize = 151_669;
const FIRST_RESERVED_ID: usize = TOKENIZER_VOCAB;
const LAST_RESERVED_ID: usize = LM_HEAD_VOCAB - 1;
const RESERVED_TAIL_ROWS: usize = LM_HEAD_VOCAB - TOKENIZER_VOCAB;
const DETERMINISTIC_SAMPLER: &str = "greedy_argmax_lowest_token_id_tie_break";

#[derive(Clone, Debug, Eq, PartialEq, Serialize)]
struct SourceTokenizerTemplateAuthority {
    model_id: String,
    source_repository: String,
    source_revision: String,
    source_config_sha256: String,
    source_tokenizer_sha256: String,
    source_tokenizer_config_sha256: String,
    source_chat_template_sha256: String,
    source_generation_config_sha256: String,
    tokenizer_addressable_vocab_size: usize,
    lm_head_vocab_size: usize,
    first_reserved_lm_head_id: usize,
    reserved_lm_head_tail_rows: usize,
}

impl SourceTokenizerTemplateAuthority {
    fn exact() -> Self {
        Self {
            model_id: MODEL_ID.into(),
            source_repository: SOURCE_REPOSITORY.into(),
            source_revision: SOURCE_REVISION.into(),
            source_config_sha256: SOURCE_CONFIG_SHA256.into(),
            source_tokenizer_sha256: SOURCE_TOKENIZER_SHA256.into(),
            source_tokenizer_config_sha256: SOURCE_TOKENIZER_CONFIG_SHA256.into(),
            source_chat_template_sha256: SOURCE_CHAT_TEMPLATE_SHA256.into(),
            source_generation_config_sha256: SOURCE_GENERATION_CONFIG_SHA256.into(),
            tokenizer_addressable_vocab_size: TOKENIZER_VOCAB,
            lm_head_vocab_size: LM_HEAD_VOCAB,
            first_reserved_lm_head_id: FIRST_RESERVED_ID,
            reserved_lm_head_tail_rows: RESERVED_TAIL_ROWS,
        }
    }

    fn validate_exact(&self) -> Result<(), String> {
        if self != &Self::exact() {
            return Err(
                "source tokenizer/template authority drifted from the admitted Qwen80 binding"
                    .into(),
            );
        }
        Ok(())
    }

    fn validate_feedback_id(&self, token_id: usize) -> Result<(), String> {
        self.validate_exact()?;
        if token_id >= self.tokenizer_addressable_vocab_size {
            return Err("reserved/non-token lm-head ID cannot enter tokenizer, embedding, or state feedback".into());
        }
        Ok(())
    }
}

#[derive(Clone, Debug, Serialize)]
struct FileEvidence {
    path: String,
    bytes: usize,
    document_sha256: String,
}

#[derive(Clone, Debug, Serialize)]
struct TerminalHeadContractAuthority {
    contract: FileEvidence,
    schema: String,
    status: String,
    unsealed_preimage_sha256: String,
    source_tokenizer_template_authority: SourceTokenizerTemplateAuthority,
    contract_is_component_only: bool,
    contract_authorizes_device_now: bool,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum FeedbackPhase {
    Fresh,
    SnapshotTaken,
    FeedbackWritten,
    RolledBack,
}

/// Borrowed active + rollback arrays supplied by one logical session.  This
/// target never allocates, shares, or retains them.  Rust's two exclusive
/// mutable borrows make the active/rollback arrays non-aliasing at the API
/// boundary without claiming anything about a future device allocation.
struct CallerOwnedSessionFeedback<'a> {
    session_id: &'a str,
    active_token_ids: &'a mut [u32],
    rollback_token_ids: &'a mut [u32],
    next_token_index: usize,
}

impl<'a> CallerOwnedSessionFeedback<'a> {
    fn validate(&self) -> Result<(), String> {
        if self.session_id.is_empty() {
            return Err("logical session ID must be non-empty".into());
        }
        if self.active_token_ids.is_empty()
            || self.active_token_ids.len() != self.rollback_token_ids.len()
        {
            return Err("caller-owned active and rollback feedback buffers must be non-empty and equal-length".into());
        }
        if self.next_token_index >= self.active_token_ids.len() {
            return Err(
                "next-token feedback index is outside the caller-owned session buffer".into(),
            );
        }
        Ok(())
    }
}

/// A future terminal head owns the actual all-row computation.  This wrapper
/// accepts only the resulting full, *already tail-masked* vector and proves
/// its shape + mask before the sampler can observe it.
struct TailMaskedLogits<'a> {
    values: &'a [f32],
}

impl<'a> TailMaskedLogits<'a> {
    fn new(values: &'a [f32]) -> Result<Self, String> {
        if values.len() != LM_HEAD_VOCAB {
            return Err(format!(
                "future sampler input must be a full [{LM_HEAD_VOCAB}] lm-head vector, got [{}]",
                values.len()
            ));
        }
        for (offset, &logit) in values[FIRST_RESERVED_ID..].iter().enumerate() {
            if !is_negative_infinity(logit) {
                return Err(format!(
                    "reserved-tail logit {} was observable by the sampler instead of negative infinity",
                    FIRST_RESERVED_ID + offset
                ));
            }
        }
        Ok(Self { values })
    }

    fn deterministic_sample(&self) -> Result<usize, String> {
        let mut selected: Option<(usize, f32)> = None;
        for (token_id, &logit) in self.values[..TOKENIZER_VOCAB].iter().enumerate() {
            if logit.is_nan() || logit == f32::INFINITY {
                return Err(format!(
                    "sampler rejects NaN or positive-infinity logit at tokenizer ID {token_id}"
                ));
            }
            if is_negative_infinity(logit) {
                continue;
            }
            match selected {
                None => selected = Some((token_id, logit)),
                Some((_, current)) if logit > current => selected = Some((token_id, logit)),
                // Iteration is ascending token ID, so retaining the first
                // equal maximum implements the required lowest-ID tie break.
                Some(_) => {}
            }
        }
        let token_id = selected
            .map(|(token_id, _)| token_id)
            .ok_or("tail-masked logit vector has no finite tokenizer-addressable candidate")?;
        if token_id >= TOKENIZER_VOCAB {
            return Err("deterministic sampler produced a reserved token ID".into());
        }
        Ok(token_id)
    }
}

fn is_negative_infinity(value: f32) -> bool {
    value.is_infinite() && value.is_sign_negative()
}

/// Stateful feedback transaction over caller-owned session storage.  A future
/// runtime must use an equivalent transaction around each candidate decode
/// step; this static preflight does not issue an inference step itself.
struct SamplerFeedbackTransaction<'a> {
    authority: SourceTokenizerTemplateAuthority,
    buffers: CallerOwnedSessionFeedback<'a>,
    phase: FeedbackPhase,
    active_preimage_sha256: Option<String>,
    sampled_token_id: Option<usize>,
}

impl<'a> SamplerFeedbackTransaction<'a> {
    fn begin(
        authority: SourceTokenizerTemplateAuthority,
        buffers: CallerOwnedSessionFeedback<'a>,
    ) -> Result<Self, String> {
        authority.validate_exact()?;
        buffers.validate()?;
        Ok(Self {
            authority,
            buffers,
            phase: FeedbackPhase::Fresh,
            active_preimage_sha256: None,
            sampled_token_id: None,
        })
    }

    fn snapshot(&mut self) -> Result<(), String> {
        if self.phase != FeedbackPhase::Fresh {
            return Err(
                "feedback rollback snapshot may be taken only once before feedback write".into(),
            );
        }
        self.buffers
            .rollback_token_ids
            .copy_from_slice(self.buffers.active_token_ids);
        self.active_preimage_sha256 = Some(u32_buffer_sha256(self.buffers.active_token_ids));
        self.phase = FeedbackPhase::SnapshotTaken;
        Ok(())
    }

    fn sample_and_write(&mut self, logits: TailMaskedLogits<'_>) -> Result<usize, String> {
        if self.phase != FeedbackPhase::SnapshotTaken {
            return Err(
                "sampling and feedback require an exact active-buffer rollback snapshot first"
                    .into(),
            );
        }
        let token_id = logits.deterministic_sample()?;
        self.authority.validate_feedback_id(token_id)?;
        self.buffers.active_token_ids[self.buffers.next_token_index] = token_id as u32;
        self.sampled_token_id = Some(token_id);
        self.phase = FeedbackPhase::FeedbackWritten;
        Ok(token_id)
    }

    fn rollback_identity(&mut self) -> Result<(), String> {
        if self.phase != FeedbackPhase::FeedbackWritten {
            return Err("feedback rollback requires exactly one sampled feedback write".into());
        }
        self.buffers
            .active_token_ids
            .copy_from_slice(self.buffers.rollback_token_ids);
        let expected = self
            .active_preimage_sha256
            .as_deref()
            .ok_or("rollback preimage was not retained")?;
        if u32_buffer_sha256(self.buffers.active_token_ids) != expected {
            return Err(
                "feedback rollback did not restore the active session buffer byte-for-byte".into(),
            );
        }
        self.phase = FeedbackPhase::RolledBack;
        Ok(())
    }

    fn sampled_token_id(&self) -> Option<usize> {
        self.sampled_token_id
    }

    fn phase(&self) -> FeedbackPhase {
        self.phase
    }
}

fn u32_buffer_sha256(values: &[u32]) -> String {
    let mut digest = Sha256::new();
    for value in values {
        digest.update(value.to_le_bytes());
    }
    format!("{:x}", digest.finalize())
}

#[derive(Serialize)]
struct FutureInputReceiptContract {
    schema: &'static str,
    required_status: &'static str,
    producer: &'static str,
    source_tokenizer_template_authority: SourceTokenizerTemplateAuthority,
    exact_logit_shape: [usize; 1],
    exact_tail_mask: TailMaskContract,
    sampling_policy: &'static str,
    required_order: Vec<&'static str>,
    rejected_inputs: Vec<&'static str>,
}

#[derive(Serialize)]
struct TailMaskContract {
    first_reserved_id: usize,
    last_reserved_id: usize,
    reserved_rows: usize,
    required_value: &'static str,
}

#[derive(Serialize)]
struct FutureOutputReceiptContract {
    schema: &'static str,
    required_status: &'static str,
    component_only: bool,
    required_fields: Vec<&'static str>,
    caller_owned_session_buffer: CallerOwnedBufferContract,
    completion_boundary: Vec<&'static str>,
}

#[derive(Serialize)]
struct CallerOwnedBufferContract {
    active_buffer_element_type: &'static str,
    rollback_buffer_element_type: &'static str,
    active_and_rollback_length_relation: &'static str,
    session_ownership: &'static str,
    next_token_write: &'static str,
    rollback_identity: &'static str,
}

#[derive(Serialize)]
struct RejectionReport {
    wrong_logit_shape_rejected: bool,
    non_negative_infinity_reserved_tail_rejected: bool,
    nan_or_positive_infinity_tokenizer_logit_rejected: bool,
    no_eligible_token_rejected: bool,
    deterministic_lowest_id_tie_break_verified: bool,
    feedback_before_snapshot_rejected: bool,
    reserved_feedback_id_rejected: bool,
    rollback_restores_active_buffer_byte_identically: bool,
    source_tokenizer_template_authority_drift_rejected: bool,
    terminal_head_contract_drift_rejected: bool,
}

#[derive(Serialize)]
struct Report {
    schema: &'static str,
    status: &'static str,
    component_only: bool,
    terminal_head_contract_authority: TerminalHeadContractAuthority,
    future_input_receipt_contract: FutureInputReceiptContract,
    future_output_receipt_contract: FutureOutputReceiptContract,
    focused_rejection_tests: RejectionReport,
    artifact_scan_performed: bool,
    tokenizer_or_template_render_performed: bool,
    metal_or_gpu_performed: bool,
    lease_or_registry_mutation_performed: bool,
    runtime_watcher_or_server_started: bool,
    decoder_token_hcli_tps_or_tg_claimed: bool,
    claim_boundary: Vec<&'static str>,
    unsealed_preimage_sha256: String,
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

fn regular_json(path: &Path, label: &str) -> Result<(FileEvidence, Value), String> {
    let metadata = fs::symlink_metadata(path)
        .map_err(|error| format!("{label} metadata failed at {}: {error}", path.display()))?;
    if !metadata.file_type().is_file() || metadata.file_type().is_symlink() {
        return Err(format!("{label} must be a regular non-symlink file"));
    }
    let bytes = fs::read(path).map_err(|error| format!("{label} read failed: {error}"))?;
    let document = serde_json::from_slice(&bytes)
        .map_err(|error| format!("{label} is invalid JSON: {error}"))?;
    Ok((
        FileEvidence {
            path: path.display().to_string(),
            bytes: bytes.len(),
            document_sha256: sha256_hex(&bytes),
        },
        document,
    ))
}

fn object<'a>(
    value: &'a Value,
    field: &str,
    label: &str,
) -> Result<&'a Map<String, Value>, String> {
    value
        .as_object()
        .and_then(|root| root.get(field))
        .and_then(Value::as_object)
        .ok_or_else(|| format!("{label} missing object {field:?}"))
}

fn required_string<'a>(
    object: &'a Map<String, Value>,
    field: &str,
    label: &str,
) -> Result<&'a str, String> {
    object
        .get(field)
        .and_then(Value::as_str)
        .ok_or_else(|| format!("{label} missing string {field:?}"))
}

fn required_string_eq(
    object: &Map<String, Value>,
    field: &str,
    expected: &str,
    label: &str,
) -> Result<(), String> {
    let actual = required_string(object, field, label)?;
    if actual != expected {
        return Err(format!("{label} {field:?} drifted"));
    }
    Ok(())
}

fn required_usize(object: &Map<String, Value>, field: &str, label: &str) -> Result<usize, String> {
    object
        .get(field)
        .and_then(Value::as_u64)
        .and_then(|value| usize::try_from(value).ok())
        .ok_or_else(|| format!("{label} missing usize {field:?}"))
}

fn required_bool_eq(
    object: &Map<String, Value>,
    field: &str,
    expected: bool,
    label: &str,
) -> Result<(), String> {
    if object.get(field).and_then(Value::as_bool) != Some(expected) {
        return Err(format!("{label} {field:?} drifted"));
    }
    Ok(())
}

fn parse_source_authority(
    binding: &Map<String, Value>,
) -> Result<SourceTokenizerTemplateAuthority, String> {
    let authority = SourceTokenizerTemplateAuthority {
        model_id: required_string(binding, "model_id", "terminal source binding")?.into(),
        source_repository: required_string(
            binding,
            "source_repository",
            "terminal source binding",
        )?
        .into(),
        source_revision: required_string(binding, "source_revision", "terminal source binding")?
            .into(),
        source_config_sha256: required_string(
            binding,
            "source_config_sha256",
            "terminal source binding",
        )?
        .into(),
        source_tokenizer_sha256: required_string(
            binding,
            "source_tokenizer_sha256",
            "terminal source binding",
        )?
        .into(),
        source_tokenizer_config_sha256: required_string(
            binding,
            "source_tokenizer_config_sha256",
            "terminal source binding",
        )?
        .into(),
        source_chat_template_sha256: required_string(
            binding,
            "source_chat_template_sha256",
            "terminal source binding",
        )?
        .into(),
        source_generation_config_sha256: required_string(
            binding,
            "source_generation_config_sha256",
            "terminal source binding",
        )?
        .into(),
        // The terminal-head contract places lm-head/tokenizer geometry beside
        // this source binding.  Keep the source/template digest binding and
        // the immutable source geometry together in the sampler authority.
        tokenizer_addressable_vocab_size: TOKENIZER_VOCAB,
        lm_head_vocab_size: LM_HEAD_VOCAB,
        first_reserved_lm_head_id: FIRST_RESERVED_ID,
        reserved_lm_head_tail_rows: RESERVED_TAIL_ROWS,
    };
    authority.validate_exact()?;
    Ok(authority)
}

fn load_terminal_head_contract(path: &Path) -> Result<TerminalHeadContractAuthority, String> {
    let (evidence, document) = regular_json(path, "terminal-head contract")?;
    let root = document
        .as_object()
        .ok_or("terminal-head contract root must be an object")?;
    required_string_eq(
        root,
        "schema",
        TERMINAL_HEAD_CONTRACT_SCHEMA,
        "terminal-head contract",
    )?;
    required_string_eq(
        root,
        "status",
        TERMINAL_HEAD_CONTRACT_STATUS,
        "terminal-head contract",
    )?;
    required_bool_eq(
        root,
        "component_contract_only",
        true,
        "terminal-head contract",
    )?;
    required_bool_eq(
        root,
        "live_artifact_scan_performed",
        false,
        "terminal-head contract",
    )?;
    required_bool_eq(
        root,
        "metal_device_or_dispatch_performed",
        false,
        "terminal-head contract",
    )?;
    required_bool_eq(
        root,
        "model_runtime_or_server_started",
        false,
        "terminal-head contract",
    )?;
    required_bool_eq(
        root,
        "hcli_execution_performed",
        false,
        "terminal-head contract",
    )?;
    required_bool_eq(
        root,
        "tps_or_tg_measurement_performed",
        false,
        "terminal-head contract",
    )?;
    let current = object(
        &document,
        "terminal_head_cpu_components",
        "terminal-head contract",
    )?;
    let authority = parse_source_authority(
        current
            .get("source_binding")
            .and_then(Value::as_object)
            .ok_or("terminal-head contract missing current source binding")?,
    )?;
    if required_usize(
        current,
        "tokenizer_addressable_vocab_size",
        "terminal current components",
    )? != TOKENIZER_VOCAB
        || required_usize(current, "lm_head_vocab_size", "terminal current components")?
            != LM_HEAD_VOCAB
        || required_usize(
            current,
            "reserved_tail_first_id",
            "terminal current components",
        )? != FIRST_RESERVED_ID
        || required_usize(
            current,
            "reserved_tail_last_id",
            "terminal current components",
        )? != LAST_RESERVED_ID
        || required_usize(current, "reserved_tail_rows", "terminal current components")?
            != RESERVED_TAIL_ROWS
    {
        return Err("terminal-head contract tail/tokenizer geometry drifted".into());
    }
    let ledger = object(
        &document,
        "future_device_dispatch_ledger",
        "terminal-head contract",
    )?;
    required_bool_eq(
        ledger,
        "dispatch_authorized_now",
        false,
        "terminal-head future ledger",
    )?;
    let preimage = required_string(root, "unsealed_preimage_sha256", "terminal-head contract")?;
    if !is_lower_sha256(preimage) {
        return Err("terminal-head contract missing a lowercase SHA-256 preimage".into());
    }
    Ok(TerminalHeadContractAuthority {
        contract: evidence,
        schema: TERMINAL_HEAD_CONTRACT_SCHEMA.into(),
        status: TERMINAL_HEAD_CONTRACT_STATUS.into(),
        unsealed_preimage_sha256: preimage.into(),
        source_tokenizer_template_authority: authority,
        contract_is_component_only: true,
        contract_authorizes_device_now: false,
    })
}

fn future_input_receipt_contract(
    authority: SourceTokenizerTemplateAuthority,
) -> FutureInputReceiptContract {
    FutureInputReceiptContract {
        schema: FUTURE_INPUT_RECEIPT_SCHEMA,
        required_status: "REQUIRED_FUTURE_SOURCE_BOUND_ALL_ROW_TAIL_MASKED_LOGIT_INPUT_NOT_CURRENTLY_AVAILABLE",
        producer: "one future full terminal-head path after all 48 source-bound Qwen80 layers",
        source_tokenizer_template_authority: authority,
        exact_logit_shape: [LM_HEAD_VOCAB],
        exact_tail_mask: TailMaskContract {
            first_reserved_id: FIRST_RESERVED_ID,
            last_reserved_id: LAST_RESERVED_ID,
            reserved_rows: RESERVED_TAIL_ROWS,
            required_value: "negative_infinity",
        },
        sampling_policy: DETERMINISTIC_SAMPLER,
        required_order: vec![
            "bind one source-bound, non-synthetic all-48-layer hidden state to terminal-head authority",
            "compute final RMSNorm and every direct-packed lm_head row [151936,2048]",
            "write negative infinity to every ID 151669..151935",
            "verify the full tail-masked [151936] vector before sampler access",
            "perform deterministic greedy argmax with lowest-token-ID tie break",
        ],
        rejected_inputs: vec![
            "selected-row or sparse logit vectors",
            "a logit vector with shape other than [151936]",
            "any finite, NaN, or positive-infinity reserved-tail logit",
            "a vector without a finite tokenizer-addressable candidate",
            "a source/tokenizer/template binding different from the terminal-head authority",
        ],
    }
}

fn future_output_receipt_contract() -> FutureOutputReceiptContract {
    FutureOutputReceiptContract {
        schema: FUTURE_OUTPUT_RECEIPT_SCHEMA,
        required_status: "REQUIRED_FUTURE_TAIL_MASKED_SAMPLER_FEEDBACK_COMPONENT_RESULT_NOT_CURRENTLY_AVAILABLE",
        component_only: true,
        required_fields: vec![
            "terminal_head_input_receipt_document_sha256",
            "terminal_head_input_receipt_seal_sha256",
            "source_tokenizer_template_authority",
            "logical_session_id",
            "active_feedback_buffer_preimage_sha256",
            "active_feedback_buffer_postwrite_sha256",
            "rollback_feedback_buffer_preimage_sha256",
            "rollback_restored_active_buffer_sha256",
            "sampled_token_id",
            "sampling_policy",
            "tail_mask_first_reserved_id",
            "tail_mask_last_reserved_id",
            "tail_mask_reserved_rows",
            "rollback_identity_verified",
            "component_only",
        ],
        caller_owned_session_buffer: CallerOwnedBufferContract {
            active_buffer_element_type: "u32 source-tokenizer IDs",
            rollback_buffer_element_type: "u32 source-tokenizer IDs",
            active_and_rollback_length_relation: "equal, non-empty logical-session arrays",
            session_ownership: "caller retains both arrays; sampler component borrows them for one transaction only",
            next_token_write: "write one validated sampled ID at the caller-supplied next-token index after snapshot",
            rollback_identity: "copy rollback array over active array and require SHA-256(active little-endian u32 bytes) equals saved active preimage",
        },
        completion_boundary: vec![
            "A result must remain component-only until a complete source-bound all-layer terminal-head path and state update are independently sealed.",
            "The receipt cannot assert decoder completion, a generated token, HCLI, BASE_TRUE_TPS, TG, capability, server, or tournament evidence.",
        ],
    }
}

fn terminal_authority_fixture() -> TerminalHeadContractAuthority {
    TerminalHeadContractAuthority {
        contract: FileEvidence {
            path: "/fixture/terminal-head-contract.json".into(),
            bytes: 1,
            document_sha256: "a".repeat(64),
        },
        schema: TERMINAL_HEAD_CONTRACT_SCHEMA.into(),
        status: TERMINAL_HEAD_CONTRACT_STATUS.into(),
        unsealed_preimage_sha256: "b".repeat(64),
        source_tokenizer_template_authority: SourceTokenizerTemplateAuthority::exact(),
        contract_is_component_only: true,
        contract_authorizes_device_now: false,
    }
}

fn sample_logits(entries: &[(usize, f32)]) -> Vec<f32> {
    let mut logits = vec![f32::NEG_INFINITY; LM_HEAD_VOCAB];
    for &(token_id, logit) in entries {
        logits[token_id] = logit;
    }
    logits
}

fn rejection_report() -> RejectionReport {
    let wrong_logit_shape_rejected = TailMaskedLogits::new(&vec![0.0; LM_HEAD_VOCAB - 1]).is_err();

    let mut tail_visible = sample_logits(&[(7, 1.0)]);
    tail_visible[FIRST_RESERVED_ID] = 0.0;
    let non_negative_infinity_reserved_tail_rejected =
        TailMaskedLogits::new(&tail_visible).is_err();

    let nan_or_positive_infinity_tokenizer_logit_rejected = {
        let nan = sample_logits(&[(7, f32::NAN)]);
        let positive_infinity = sample_logits(&[(7, f32::INFINITY)]);
        TailMaskedLogits::new(&nan)
            .and_then(|logits| logits.deterministic_sample())
            .is_err()
            && TailMaskedLogits::new(&positive_infinity)
                .and_then(|logits| logits.deterministic_sample())
                .is_err()
    };

    let no_eligible_token_rejected = TailMaskedLogits::new(&sample_logits(&[]))
        .and_then(|logits| logits.deterministic_sample())
        .is_err();

    let deterministic_lowest_id_tie_break_verified =
        TailMaskedLogits::new(&sample_logits(&[(19, 4.0), (7, 4.0), (1, 3.0)]))
            .and_then(|logits| logits.deterministic_sample())
            == Ok(7);

    let mut active = [11_u32, 12, 13];
    let mut rollback = [0_u32; 3];
    let mut transaction = SamplerFeedbackTransaction::begin(
        SourceTokenizerTemplateAuthority::exact(),
        CallerOwnedSessionFeedback {
            session_id: "fixture-session",
            active_token_ids: &mut active,
            rollback_token_ids: &mut rollback,
            next_token_index: 2,
        },
    )
    .expect("fixture transaction must be valid");
    let feedback_before_snapshot_rejected = transaction
        .sample_and_write(
            TailMaskedLogits::new(&sample_logits(&[(9, 1.0)])).expect("fixture logits"),
        )
        .is_err();
    transaction.snapshot().expect("fixture snapshot");
    let reserved_feedback_id_rejected = transaction
        .authority
        .validate_feedback_id(FIRST_RESERVED_ID)
        .is_err();
    transaction
        .sample_and_write(
            TailMaskedLogits::new(&sample_logits(&[(9, 1.0)])).expect("fixture logits"),
        )
        .expect("fixture feedback write");
    transaction.rollback_identity().expect("fixture rollback");
    let phase = transaction.phase();
    let sampled = transaction.sampled_token_id();
    drop(transaction);
    let rollback_restores_active_buffer_byte_identically = active == [11, 12, 13]
        && rollback == [11, 12, 13]
        && phase == FeedbackPhase::RolledBack
        && sampled == Some(9);

    let mut drifted = SourceTokenizerTemplateAuthority::exact();
    drifted.source_chat_template_sha256 = "c".repeat(64);
    let source_tokenizer_template_authority_drift_rejected = SamplerFeedbackTransaction::begin(
        drifted,
        CallerOwnedSessionFeedback {
            session_id: "drifted",
            active_token_ids: &mut [1_u32],
            rollback_token_ids: &mut [0_u32],
            next_token_index: 0,
        },
    )
    .is_err();

    let terminal_head_contract_drift_rejected = {
        let mut authority = terminal_authority_fixture();
        authority.status = "not-the-terminal-contract".into();
        authority.status != TERMINAL_HEAD_CONTRACT_STATUS
            || authority.contract_authorizes_device_now
    };

    RejectionReport {
        wrong_logit_shape_rejected,
        non_negative_infinity_reserved_tail_rejected,
        nan_or_positive_infinity_tokenizer_logit_rejected,
        no_eligible_token_rejected,
        deterministic_lowest_id_tie_break_verified,
        feedback_before_snapshot_rejected,
        reserved_feedback_id_rejected,
        rollback_restores_active_buffer_byte_identically,
        source_tokenizer_template_authority_drift_rejected,
        terminal_head_contract_drift_rejected,
    }
}

fn all_rejections_pass(report: &RejectionReport) -> bool {
    report.wrong_logit_shape_rejected
        && report.non_negative_infinity_reserved_tail_rejected
        && report.nan_or_positive_infinity_tokenizer_logit_rejected
        && report.no_eligible_token_rejected
        && report.deterministic_lowest_id_tie_break_verified
        && report.feedback_before_snapshot_rejected
        && report.reserved_feedback_id_rejected
        && report.rollback_restores_active_buffer_byte_identically
        && report.source_tokenizer_template_authority_drift_rejected
        && report.terminal_head_contract_drift_rejected
}

fn report(authority: TerminalHeadContractAuthority) -> Report {
    Report {
        schema: SCHEMA,
        status: STATUS,
        component_only: true,
        future_input_receipt_contract: future_input_receipt_contract(
            authority.source_tokenizer_template_authority.clone(),
        ),
        future_output_receipt_contract: future_output_receipt_contract(),
        terminal_head_contract_authority: authority,
        focused_rejection_tests: rejection_report(),
        artifact_scan_performed: false,
        tokenizer_or_template_render_performed: false,
        metal_or_gpu_performed: false,
        lease_or_registry_mutation_performed: false,
        runtime_watcher_or_server_started: false,
        decoder_token_hcli_tps_or_tg_claimed: false,
        claim_boundary: vec![
            "This preflight reads one explicit terminal-head contract JSON only. It does not open a tensor artifact, source sidecar, tokenizer file, template file, or packed payload.",
            "The synthetic unit-test logits establish sampler and rollback ABI behavior only; they are not a Qwen80 terminal-head result or generated token.",
            "No Metal/GPU, registry, lease, runtime, watcher, server, decoder, HCLI, TPS, TG, capability, or tournament action occurs here.",
            "A future full-path receipt must bind a source-bound all-48-layer terminal-head logit vector, prove the exact tail mask, then prove the caller-owned session feedback write and exact rollback.",
        ],
        unsealed_preimage_sha256: String::new(),
    }
}

fn write_report_create_new(path: &Path, report: &Report) -> Result<(), Box<dyn Error>> {
    let parent = path.parent().ok_or("output path has no parent")?;
    if !parent.is_dir() {
        return Err(format!("output parent is missing: {}", parent.display()).into());
    }
    let mut output = OpenOptions::new().write(true).create_new(true).open(path)?;
    output.write_all(&serde_json::to_vec_pretty(report)?)?;
    output.write_all(b"\n")?;
    output.sync_all()?;
    Ok(())
}

struct Args {
    terminal_head_contract: PathBuf,
    out: PathBuf,
}

fn usage() -> &'static str {
    "usage: ascension_qwen80_sampler_feedback_component_preflight \\\n+--terminal-head-contract ABSOLUTE_PATH --out ABSOLUTE_NEW_PATH"
}

fn parse_args() -> Result<Args, Box<dyn Error>> {
    let mut terminal_head_contract = None;
    let mut out = None;
    let mut values = env::args().skip(1);
    while let Some(flag) = values.next() {
        let value = values
            .next()
            .ok_or_else(|| format!("missing value after {flag}; {}", usage()))?;
        let target = PathBuf::from(value);
        match flag.as_str() {
            "--terminal-head-contract" => {
                if terminal_head_contract.replace(target).is_some() {
                    return Err("--terminal-head-contract repeated".into());
                }
            }
            "--out" => {
                if out.replace(target).is_some() {
                    return Err("--out repeated".into());
                }
            }
            _ => return Err(format!("unsupported option {flag:?}; {}", usage()).into()),
        }
    }
    let args = Args {
        terminal_head_contract: terminal_head_contract
            .ok_or_else(|| format!("missing --terminal-head-contract; {}", usage()))?,
        out: out.ok_or_else(|| format!("missing --out; {}", usage()))?,
    };
    if !args.terminal_head_contract.is_absolute() || !args.out.is_absolute() {
        return Err("all paths must be absolute".into());
    }
    Ok(args)
}

fn run(args: Args) -> Result<(), Box<dyn Error>> {
    let authority = load_terminal_head_contract(&args.terminal_head_contract)?;
    let rejection_tests = rejection_report();
    if !all_rejections_pass(&rejection_tests) {
        return Err("sampler/feedback preflight rejection tests did not all pass".into());
    }
    let mut output = report(authority);
    output.focused_rejection_tests = rejection_tests;
    output.unsealed_preimage_sha256 = sha256_hex(&serde_json::to_vec(&output)?);
    write_report_create_new(&args.out, &output)
}

fn main() {
    match parse_args().and_then(run) {
        Ok(()) => {}
        Err(error) => {
            eprintln!("ascension_qwen80_sampler_feedback_component_preflight: {error}");
            std::process::exit(2);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;
    use tempfile::tempdir;

    #[test]
    fn exact_full_tail_masked_shape_and_lowest_id_tie_break_are_required() {
        assert_eq!(LM_HEAD_VOCAB, 151_936);
        assert_eq!(TOKENIZER_VOCAB, 151_669);
        assert_eq!(RESERVED_TAIL_ROWS, 267);
        let logits = sample_logits(&[(91, 6.0), (7, 6.0), (1, 5.0)]);
        assert_eq!(
            TailMaskedLogits::new(&logits)
                .unwrap()
                .deterministic_sample(),
            Ok(7)
        );
        assert!(TailMaskedLogits::new(&logits[..LM_HEAD_VOCAB - 1]).is_err());
    }

    #[test]
    fn reserved_tail_and_nonfinite_tokenizer_inputs_are_fail_closed() {
        let mut tail_visible = sample_logits(&[(7, 1.0)]);
        tail_visible[LAST_RESERVED_ID] = -12.0;
        assert!(TailMaskedLogits::new(&tail_visible).is_err());

        for invalid in [f32::NAN, f32::INFINITY] {
            let logits = sample_logits(&[(7, invalid)]);
            assert!(TailMaskedLogits::new(&logits)
                .and_then(|logits| logits.deterministic_sample())
                .is_err());
        }
        assert!(TailMaskedLogits::new(&sample_logits(&[]))
            .and_then(|logits| logits.deterministic_sample())
            .is_err());
    }

    #[test]
    fn caller_owned_feedback_requires_snapshot_and_rolls_back_byte_identically() {
        let mut active = [100_u32, 101, 102, 103];
        let mut rollback = [0_u32; 4];
        let preimage = u32_buffer_sha256(&active);
        let mut transaction = SamplerFeedbackTransaction::begin(
            SourceTokenizerTemplateAuthority::exact(),
            CallerOwnedSessionFeedback {
                session_id: "logical-session-17",
                active_token_ids: &mut active,
                rollback_token_ids: &mut rollback,
                next_token_index: 3,
            },
        )
        .unwrap();
        let logits = sample_logits(&[(33, 8.0)]);
        assert!(transaction
            .sample_and_write(TailMaskedLogits::new(&logits).unwrap())
            .is_err());
        transaction.snapshot().unwrap();
        assert_eq!(
            transaction
                .sample_and_write(TailMaskedLogits::new(&logits).unwrap())
                .unwrap(),
            33
        );
        assert_eq!(transaction.buffers.active_token_ids, &[100, 101, 102, 33]);
        transaction.rollback_identity().unwrap();
        let phase = transaction.phase();
        drop(transaction);
        assert_eq!(active, [100, 101, 102, 103]);
        assert_eq!(rollback, [100, 101, 102, 103]);
        assert_eq!(u32_buffer_sha256(&active), preimage);
        assert_eq!(phase, FeedbackPhase::RolledBack);
    }

    #[test]
    fn source_authority_and_reserved_feedback_boundary_are_exact() {
        SourceTokenizerTemplateAuthority::exact()
            .validate_feedback_id(TOKENIZER_VOCAB - 1)
            .unwrap();
        assert!(SourceTokenizerTemplateAuthority::exact()
            .validate_feedback_id(FIRST_RESERVED_ID)
            .is_err());
        let mut drifted = SourceTokenizerTemplateAuthority::exact();
        drifted.source_tokenizer_sha256 = "f".repeat(64);
        assert!(drifted.validate_exact().is_err());
    }

    fn terminal_contract_fixture() -> Value {
        json!({
            "schema": TERMINAL_HEAD_CONTRACT_SCHEMA,
            "status": TERMINAL_HEAD_CONTRACT_STATUS,
            "component_contract_only": true,
            "live_artifact_scan_performed": false,
            "metal_device_or_dispatch_performed": false,
            "model_runtime_or_server_started": false,
            "hcli_execution_performed": false,
            "tps_or_tg_measurement_performed": false,
            "terminal_head_cpu_components": {
                "source_binding": {
                    "model_id": MODEL_ID,
                    "source_repository": SOURCE_REPOSITORY,
                    "source_revision": SOURCE_REVISION,
                    "source_config_sha256": SOURCE_CONFIG_SHA256,
                    "source_tokenizer_sha256": SOURCE_TOKENIZER_SHA256,
                    "source_tokenizer_config_sha256": SOURCE_TOKENIZER_CONFIG_SHA256,
                    "source_chat_template_sha256": SOURCE_CHAT_TEMPLATE_SHA256,
                    "source_generation_config_sha256": SOURCE_GENERATION_CONFIG_SHA256,
                    "tokenizer_addressable_vocab_size": TOKENIZER_VOCAB,
                    "lm_head_vocab_size": LM_HEAD_VOCAB,
                    "first_reserved_lm_head_id": FIRST_RESERVED_ID,
                    "reserved_lm_head_tail_rows": RESERVED_TAIL_ROWS
                },
                "tokenizer_addressable_vocab_size": TOKENIZER_VOCAB,
                "lm_head_vocab_size": LM_HEAD_VOCAB,
                "reserved_tail_first_id": FIRST_RESERVED_ID,
                "reserved_tail_last_id": LAST_RESERVED_ID,
                "reserved_tail_rows": RESERVED_TAIL_ROWS
            },
            "future_device_dispatch_ledger": {"dispatch_authorized_now": false},
            "unsealed_preimage_sha256": "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa"
        })
    }

    #[test]
    fn external_terminal_head_contract_is_required_and_cannot_authorize_device() {
        let directory = tempdir().unwrap();
        let path = directory.path().join("terminal-contract.json");
        fs::write(
            &path,
            serde_json::to_vec(&terminal_contract_fixture()).unwrap(),
        )
        .unwrap();
        let authority = load_terminal_head_contract(&path).unwrap();
        assert_eq!(authority.schema, TERMINAL_HEAD_CONTRACT_SCHEMA);
        assert!(!authority.contract_authorizes_device_now);

        let mut drifted = terminal_contract_fixture();
        drifted["future_device_dispatch_ledger"]["dispatch_authorized_now"] = json!(true);
        fs::write(&path, serde_json::to_vec(&drifted).unwrap()).unwrap();
        assert!(load_terminal_head_contract(&path).is_err());
    }

    #[test]
    fn report_is_explicitly_prepared_incomplete_and_nonpromotable() {
        let report = report(terminal_authority_fixture());
        assert_eq!(report.schema, SCHEMA);
        assert_eq!(report.status, STATUS);
        assert!(report.component_only);
        assert!(!report.artifact_scan_performed);
        assert!(!report.metal_or_gpu_performed);
        assert!(!report.decoder_token_hcli_tps_or_tg_claimed);
        assert!(all_rejections_pass(&report.focused_rejection_tests));
        assert_eq!(
            report.future_input_receipt_contract.exact_logit_shape,
            [LM_HEAD_VOCAB]
        );
        assert_eq!(
            report
                .future_input_receipt_contract
                .exact_tail_mask
                .reserved_rows,
            RESERVED_TAIL_ROWS
        );
    }
}

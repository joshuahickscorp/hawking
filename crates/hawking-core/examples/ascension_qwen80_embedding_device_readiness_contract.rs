//! CPU-only Qwen3-Coder-Next embedding device-readiness contract.
//!
//! This standalone ledger binds the admitted direct-packed embedding geometry,
//! source tokenizer/chat-template namespace, and the current CPU row/tail
//! guard to the evidence a future device gather must retain.  It reads caller
//! supplied metadata only: it does not open an artifact, create Metal work,
//! invoke a runtime/watcher/server/HCLI endpoint, or measure TPS/TG.
//!
//! The current CPU oracle is a useful boundary: a source-addressable ID below
//! 151669 may select one `model.embed_tokens.weight [151936,2048]` row, while
//! the 267 source-reserved rows must be rejected before a payload access.  It
//! is deliberately not a device gather, decoder, token, or fallback path.
//! Therefore `--current-evidence` is intentionally INCOMPLETE.

use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::collections::{BTreeMap, BTreeSet};
use std::env;
use std::error::Error;
use std::fs;
use std::path::{Path, PathBuf};

const INPUT_SCHEMA: &str = "hawking.ascension.qwen80_embedding_device_readiness_input.v1";
const RESULT_SCHEMA: &str = "hawking.ascension.qwen80_embedding_device_readiness_result.v1";
const DEVICE_ROW_SCHEMA: &str = "hawking.ascension.qwen80_embedding_device_row_parity_ledger.v1";
const TAIL_REFUSAL_SCHEMA: &str =
    "hawking.ascension.qwen80_embedding_device_tail_refusal_ledger.v1";
const CPU_GUARD_SCHEMA: &str = "hawking.ascension.qwen80_direct_packed_embedding_cpu_guard.v1";
const CPU_GUARD_STATUS: &str =
    "CURRENT_QWEN80_DIRECT_PACKED_EMBEDDING_CPU_ROW_AND_TAIL_GUARD_COMPONENT_ONLY";
const TOKENIZER_CONTRACT_SCHEMA: &str =
    "hawking.ascension.qwen80_tokenizer_sampler_handoff_contract.v1";
const TOKENIZER_CONTRACT_STATUS: &str =
    "EARNED_SOURCE_BOUND_TOKENIZER_TEMPLATE_SAMPLER_HANDOFF_COMPONENT_NOT_RUNTIME_OR_TOKEN";
const TOKENIZER_CONTRACT_DOCUMENT_SHA256: &str =
    "e152b21d9eae43e7039f9d646412b2806b8f07d3d3c7ea932ab281dc6c9a0792";
const TOKENIZER_CONTRACT_UNSEALED_PREIMAGE_SHA256: &str =
    "5c2f66487c7a4fb387806bb9439259eb62c86f33b1e30ca4dac701ee38ac164c";
const MODEL_ID: &str = "Qwen3-Coder-Next-80B";
const MODEL_KEY: &str = "qwen80";
const SOURCE_REPOSITORY: &str = "Qwen/Qwen3-Coder-Next";
const SOURCE_REVISION: &str = "a7fbcb5c0e12d62a448eaa0e260346bf5dcc0feb";
const MANIFEST_SCHEMA: &str = "hawking.ascension.qwen80_complete_binary_gravity.v1";
const MANIFEST_SEAL: &str = "14cf6c4d17086dabc54b53b4dd28b9f6551ef06c6d8bf4ee8453d775d0f6817b";
const ADMISSION_RECEIPT_SEAL: &str =
    "939b41322363da3db774a2530b207bf380ed641d23cae671fc6438c0eecbf628";
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
const COMPLETE_TENSOR_COUNT: usize = 74_391;
const COMPLETE_PAYLOAD_BYTES: u64 = 11_207_187_116;
const EMBEDDING_TENSOR: &str = "model.embed_tokens.weight";
const EMBEDDING_ROWS: usize = 151_936;
const HIDDEN: usize = 2_048;
const GROUP_SIZE: usize = 128;
const TOKENIZER_VOCAB: u32 = 151_669;
const FIRST_RESERVED_ID: u32 = TOKENIZER_VOCAB;
const LAST_RESERVED_ID: u32 = EMBEDDING_ROWS as u32 - 1;
const RESERVED_TAIL_ROWS: u32 = EMBEDDING_ROWS as u32 - TOKENIZER_VOCAB;
const EMBEDDING_BUFFER_BYTES: usize = HIDDEN * std::mem::size_of::<f32>();
const CPU_ORACLE_API: &str = "Qwen80CompleteArtifactCatalog::execute_embedding_lookup_cpu_oracle";
const CPU_ROW_API: &str =
    "Qwen80CpuPackedTensor::row(..., Qwen80CpuPackedReadMode::StreamingDirect)";
const CPU_ORACLE_BOUNDARY: &str = "source-token id below the exact 151669-token namespace -> one direct-packed model.embed_tokens.weight row [2048]; no native gather, layer, decoder, generation, HCLI, or TPS execution";
const MAX_ROW_PARITY_TOLERANCE: f64 = 1.0e-3;

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
struct SourceBinding {
    model_id: String,
    model_key: String,
    source_repository: String,
    source_revision: String,
    manifest_schema: String,
    manifest_seal_sha256: String,
    admission_receipt_seal_sha256: String,
    source_config_sha256: String,
    source_tokenizer_sha256: String,
    source_tokenizer_config_sha256: String,
    source_chat_template_sha256: String,
    source_generation_config_sha256: String,
    complete_tensor_count: usize,
    complete_payload_bytes: u64,
}

impl SourceBinding {
    fn exact() -> Self {
        Self {
            model_id: MODEL_ID.into(),
            model_key: MODEL_KEY.into(),
            source_repository: SOURCE_REPOSITORY.into(),
            source_revision: SOURCE_REVISION.into(),
            manifest_schema: MANIFEST_SCHEMA.into(),
            manifest_seal_sha256: MANIFEST_SEAL.into(),
            admission_receipt_seal_sha256: ADMISSION_RECEIPT_SEAL.into(),
            source_config_sha256: SOURCE_CONFIG_SHA256.into(),
            source_tokenizer_sha256: SOURCE_TOKENIZER_SHA256.into(),
            source_tokenizer_config_sha256: SOURCE_TOKENIZER_CONFIG_SHA256.into(),
            source_chat_template_sha256: SOURCE_CHAT_TEMPLATE_SHA256.into(),
            source_generation_config_sha256: SOURCE_GENERATION_CONFIG_SHA256.into(),
            complete_tensor_count: COMPLETE_TENSOR_COUNT,
            complete_payload_bytes: COMPLETE_PAYLOAD_BYTES,
        }
    }

    fn validate_exact(&self, label: &str) -> Result<(), String> {
        if self != &Self::exact() {
            return Err(format!(
                "{label} source/config/tokenizer/artifact binding drifted"
            ));
        }
        Ok(())
    }
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
struct EmbeddingAbi {
    tensor_name: String,
    shape: Vec<usize>,
    direct_packed_group_size: usize,
    decoded_output_element_type: String,
}

impl EmbeddingAbi {
    fn exact() -> Self {
        Self {
            tensor_name: EMBEDDING_TENSOR.into(),
            shape: vec![EMBEDDING_ROWS, HIDDEN],
            direct_packed_group_size: GROUP_SIZE,
            decoded_output_element_type: "f32".into(),
        }
    }

    fn validate_exact(&self, label: &str) -> Result<(), String> {
        if self != &Self::exact() {
            return Err(format!(
                "{label} embedding ABI must remain model.embed_tokens.weight [151936,2048] direct-packed group128 -> f32"
            ));
        }
        Ok(())
    }
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
struct TokenizerTemplateTailBinding {
    schema: String,
    status: String,
    document_sha256: String,
    unsealed_preimage_sha256: String,
    source_binding: SourceBinding,
    bounded_template_mode: String,
    tokenizer_addressable_vocab_size: u32,
    embedding_rows: u32,
    first_reserved_id: u32,
    last_reserved_id: u32,
    reserved_tail_rows: u32,
    source_prompt_ids_must_be_addressable: bool,
    feedback_ids_must_be_addressable: bool,
    reserved_tail_masked_before_sampling: bool,
}

impl TokenizerTemplateTailBinding {
    fn exact() -> Self {
        Self {
            schema: TOKENIZER_CONTRACT_SCHEMA.into(),
            status: TOKENIZER_CONTRACT_STATUS.into(),
            document_sha256: TOKENIZER_CONTRACT_DOCUMENT_SHA256.into(),
            unsealed_preimage_sha256: TOKENIZER_CONTRACT_UNSEALED_PREIMAGE_SHA256.into(),
            source_binding: SourceBinding::exact(),
            bounded_template_mode: "one_user_no_system_no_tools".into(),
            tokenizer_addressable_vocab_size: TOKENIZER_VOCAB,
            embedding_rows: EMBEDDING_ROWS as u32,
            first_reserved_id: FIRST_RESERVED_ID,
            last_reserved_id: LAST_RESERVED_ID,
            reserved_tail_rows: RESERVED_TAIL_ROWS,
            source_prompt_ids_must_be_addressable: true,
            feedback_ids_must_be_addressable: true,
            reserved_tail_masked_before_sampling: true,
        }
    }

    fn validation_errors(&self) -> Vec<String> {
        let mut errors = Vec::new();
        if self != &Self::exact() {
            errors.push(
                "tokenizer/template/tail binding must match the admitted source handoff receipt exactly"
                    .into(),
            );
        }
        errors
    }
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
struct CurrentCpuRowTailGuard {
    schema: String,
    status: String,
    source_binding: SourceBinding,
    embedding: EmbeddingAbi,
    catalog_api: String,
    streaming_row_api: String,
    source_algorithm_boundary: String,
    source_token_id_must_be_below: u32,
    first_reserved_id: u32,
    last_reserved_id: u32,
    reserved_tail_rows: u32,
    admitted_catalog_required_before_row_read: bool,
    reserved_tail_rejected_before_payload_access: bool,
    valid_row_requires_exactly_2048_finite_f32_values: bool,
    cpu_oracle_only: bool,
    native_device_gather_performed: bool,
    full_decoder_or_token_performed: bool,
}

impl CurrentCpuRowTailGuard {
    fn exact() -> Self {
        Self {
            schema: CPU_GUARD_SCHEMA.into(),
            status: CPU_GUARD_STATUS.into(),
            source_binding: SourceBinding::exact(),
            embedding: EmbeddingAbi::exact(),
            catalog_api: CPU_ORACLE_API.into(),
            streaming_row_api: CPU_ROW_API.into(),
            source_algorithm_boundary: CPU_ORACLE_BOUNDARY.into(),
            source_token_id_must_be_below: TOKENIZER_VOCAB,
            first_reserved_id: FIRST_RESERVED_ID,
            last_reserved_id: LAST_RESERVED_ID,
            reserved_tail_rows: RESERVED_TAIL_ROWS,
            admitted_catalog_required_before_row_read: true,
            reserved_tail_rejected_before_payload_access: true,
            valid_row_requires_exactly_2048_finite_f32_values: true,
            cpu_oracle_only: true,
            native_device_gather_performed: false,
            full_decoder_or_token_performed: false,
        }
    }

    fn validation_errors(&self) -> Vec<String> {
        let mut errors = Vec::new();
        if self != &Self::exact() {
            errors.push(
                "current direct-packed CPU embedding row/tail guard drifted from its admitted component boundary"
                    .into(),
            );
        }
        errors
    }
}

#[derive(Clone, Copy, Debug, Deserialize, Eq, Ord, PartialEq, PartialOrd, Serialize)]
#[serde(rename_all = "snake_case")]
enum RealInputKind {
    SourceTemplatePrompt,
    ValidAutoregressiveFeedback,
}

fn required_real_input_kinds() -> BTreeSet<RealInputKind> {
    [
        RealInputKind::SourceTemplatePrompt,
        RealInputKind::ValidAutoregressiveFeedback,
    ]
    .into_iter()
    .collect()
}

#[derive(Clone, Debug, Deserialize, Serialize)]
struct SessionEmbeddingBufferProvenance {
    session_id_sha256: String,
    logical_session_generation: u64,
    allocation_receipt_sha256: String,
    token_input_buffer_id_sha256: String,
    embedding_output_buffer_id_sha256: String,
    output_offset_bytes: usize,
    output_capacity_bytes: usize,
    output_element_count: usize,
    output_row_sha256: String,
    per_session_owned: bool,
    not_aliased_to_another_logical_session: bool,
    allocated_before_gather: bool,
    retained_through_layer0_input_fence: bool,
    layer0_input_sha256: String,
}

impl SessionEmbeddingBufferProvenance {
    fn validation_errors(&self, row_index: usize, device_row_sha256: &str) -> Vec<String> {
        let mut errors = Vec::new();
        for (label, value) in [
            ("session_id", self.session_id_sha256.as_str()),
            (
                "allocation_receipt",
                self.allocation_receipt_sha256.as_str(),
            ),
            (
                "token_input_buffer",
                self.token_input_buffer_id_sha256.as_str(),
            ),
            (
                "embedding_output_buffer",
                self.embedding_output_buffer_id_sha256.as_str(),
            ),
            ("output_row", self.output_row_sha256.as_str()),
            ("layer0_input", self.layer0_input_sha256.as_str()),
        ] {
            if !is_lower_sha256(value) {
                errors.push(format!(
                    "device row {row_index} session buffer provenance has invalid {label} digest"
                ));
            }
        }
        if self.logical_session_generation == 0
            || self.output_offset_bytes % std::mem::size_of::<f32>() != 0
            || self.output_capacity_bytes != EMBEDDING_BUFFER_BYTES
            || self.output_element_count != HIDDEN
            || self.output_row_sha256 != device_row_sha256
            || self.layer0_input_sha256 != device_row_sha256
        {
            errors.push(format!(
                "device row {row_index} lacks the exact [2048] per-session embedding output buffer layout/provenance"
            ));
        }
        if !self.per_session_owned
            || !self.not_aliased_to_another_logical_session
            || !self.allocated_before_gather
            || !self.retained_through_layer0_input_fence
        {
            errors.push(format!(
                "device row {row_index} session buffer is not owned/fenced as a per-session layer-0 input"
            ));
        }
        errors
    }
}

#[derive(Clone, Debug, Deserialize, Serialize)]
struct DeviceRowParityWitness {
    schema: String,
    receipt_seal_sha256: String,
    source_binding: SourceBinding,
    embedding: EmbeddingAbi,
    embedding_artifact_sha256: String,
    tokenizer_contract_document_sha256: String,
    cpu_guard_schema: String,
    cpu_guard_status: String,
    cpu_oracle_api: String,
    input_kind: RealInputKind,
    token_id: u32,
    source_input_sha256: String,
    source_tokenizer_encoded: bool,
    source_template_or_feedback_chain_bound: bool,
    input_is_real_source_token: bool,
    synthetic_input: bool,
    fixture_only: bool,
    fallback_used: bool,
    reserved_tail_preflight_passed_before_gather: bool,
    backend: String,
    device_dispatches: usize,
    source_bound: bool,
    artifact_bound: bool,
    device_embedding_gather_performed: bool,
    cpu_row_sha256: String,
    device_row_sha256: String,
    max_abs_error: f64,
    parity_tolerance: f64,
    session_buffer: SessionEmbeddingBufferProvenance,
}

impl DeviceRowParityWitness {
    fn validation_errors(&self, row_index: usize) -> Vec<String> {
        let mut errors = Vec::new();
        if self.schema != DEVICE_ROW_SCHEMA || !is_lower_sha256(&self.receipt_seal_sha256) {
            errors.push(format!(
                "device row {row_index} is missing the exact sealed device-row ledger identity"
            ));
        }
        if let Err(error) = self
            .source_binding
            .validate_exact(&format!("device row {row_index}"))
        {
            errors.push(error);
        }
        if let Err(error) = self
            .embedding
            .validate_exact(&format!("device row {row_index}"))
        {
            errors.push(error);
        }
        if !is_lower_sha256(&self.embedding_artifact_sha256)
            || self.tokenizer_contract_document_sha256 != TOKENIZER_CONTRACT_DOCUMENT_SHA256
            || self.cpu_guard_schema != CPU_GUARD_SCHEMA
            || self.cpu_guard_status != CPU_GUARD_STATUS
            || self.cpu_oracle_api != CPU_ORACLE_API
        {
            errors.push(format!(
                "device row {row_index} does not bind the admitted artifact/tokenizer/CPU-row authority"
            ));
        }
        if self.token_id >= TOKENIZER_VOCAB {
            errors.push(format!(
                "device row {row_index} token {} is in reserved tail {}..{} and must be refused before gather",
                self.token_id, FIRST_RESERVED_ID, LAST_RESERVED_ID
            ));
        }
        if !is_lower_sha256(&self.source_input_sha256)
            || !self.source_tokenizer_encoded
            || !self.source_template_or_feedback_chain_bound
            || !self.input_is_real_source_token
            || self.synthetic_input
            || self.fixture_only
            || self.fallback_used
            || !self.reserved_tail_preflight_passed_before_gather
        {
            errors.push(format!(
                "device row {row_index} is not a real tokenizer/template/feedback input with tail refusal before gather"
            ));
        }
        if self.backend != "metal"
            || self.device_dispatches == 0
            || !self.source_bound
            || !self.artifact_bound
            || !self.device_embedding_gather_performed
        {
            errors.push(format!(
                "device row {row_index} lacks an actual source/artifact-bound device embedding gather"
            ));
        }
        if !is_lower_sha256(&self.cpu_row_sha256)
            || !is_lower_sha256(&self.device_row_sha256)
            || !self.max_abs_error.is_finite()
            || self.max_abs_error < 0.0
            || !self.parity_tolerance.is_finite()
            || self.parity_tolerance <= 0.0
            || self.parity_tolerance > MAX_ROW_PARITY_TOLERANCE
            || self.max_abs_error > self.parity_tolerance
        {
            errors.push(format!(
                "device row {row_index} lacks bounded finite CPU/device [2048] row parity"
            ));
        }
        errors.extend(
            self.session_buffer
                .validation_errors(row_index, &self.device_row_sha256),
        );
        errors
    }
}

#[derive(Clone, Debug, Deserialize, Serialize)]
struct TailRefusalDeviceWitness {
    schema: String,
    receipt_seal_sha256: String,
    source_binding: SourceBinding,
    embedding: EmbeddingAbi,
    embedding_artifact_sha256: String,
    first_reserved_id: u32,
    last_reserved_id: u32,
    reserved_tail_rows: u32,
    backend: String,
    source_bound: bool,
    artifact_bound: bool,
    device_tail_preflight_performed: bool,
    first_reserved_rejected_before_gather: bool,
    last_reserved_rejected_before_gather: bool,
    every_reserved_id_rejected_before_gather: bool,
    no_reserved_embedding_buffer_write: bool,
    no_reserved_layer0_input: bool,
    no_reserved_state_mutation: bool,
    fixture_only: bool,
    fallback_used: bool,
}

impl TailRefusalDeviceWitness {
    fn validation_errors(&self) -> Vec<String> {
        let mut errors = Vec::new();
        if self.schema != TAIL_REFUSAL_SCHEMA || !is_lower_sha256(&self.receipt_seal_sha256) {
            errors.push("tail-refusal witness lacks an exact sealed ledger identity".into());
        }
        if let Err(error) = self.source_binding.validate_exact("tail-refusal witness") {
            errors.push(error);
        }
        if let Err(error) = self.embedding.validate_exact("tail-refusal witness") {
            errors.push(error);
        }
        if !is_lower_sha256(&self.embedding_artifact_sha256)
            || self.first_reserved_id != FIRST_RESERVED_ID
            || self.last_reserved_id != LAST_RESERVED_ID
            || self.reserved_tail_rows != RESERVED_TAIL_ROWS
        {
            errors.push("tail-refusal witness embedding/tail partition drifted".into());
        }
        if self.backend != "metal"
            || !self.source_bound
            || !self.artifact_bound
            || !self.device_tail_preflight_performed
            || !self.first_reserved_rejected_before_gather
            || !self.last_reserved_rejected_before_gather
            || !self.every_reserved_id_rejected_before_gather
            || !self.no_reserved_embedding_buffer_write
            || !self.no_reserved_layer0_input
            || !self.no_reserved_state_mutation
            || self.fixture_only
            || self.fallback_used
        {
            errors.push(
                "tail-refusal witness must prove every reserved ID is refused before gather/buffer/layer0/state mutation"
                    .into(),
            );
        }
        errors
    }
}

#[derive(Clone, Debug, Deserialize, Serialize)]
struct ReadinessInput {
    schema: String,
    source_binding: SourceBinding,
    tokenizer_template_tail_binding: TokenizerTemplateTailBinding,
    current_cpu_row_tail_guard: CurrentCpuRowTailGuard,
    device_row_witnesses: Vec<DeviceRowParityWitness>,
    tail_refusal_witness: Option<TailRefusalDeviceWitness>,
}

#[derive(Serialize)]
struct RowAssessment {
    index: usize,
    input_kind: RealInputKind,
    token_id: u32,
    session_id_sha256: String,
    embedding_output_buffer_id_sha256: String,
    satisfied: bool,
    failures: Vec<String>,
}

#[derive(Serialize)]
struct ReadinessReport {
    schema: &'static str,
    status: &'static str,
    embedding_device_readiness_earned: bool,
    complete_decoder_readiness_earned: bool,
    real_gravity_server_launch_precondition_satisfied: bool,
    input_schema_valid: bool,
    source_binding_valid: bool,
    tokenizer_template_tail_binding_valid: bool,
    current_cpu_row_tail_guard_valid: bool,
    current_cpu_guard_is_device_evidence: bool,
    required_real_input_kinds: Vec<RealInputKind>,
    observed_valid_real_input_kinds: Vec<RealInputKind>,
    row_assessments: Vec<RowAssessment>,
    tail_refusal_witness_valid: bool,
    tail_refusal_failures: Vec<String>,
    contract_errors: Vec<String>,
    read_only_contract: bool,
    live_artifact_scan_performed: bool,
    metal_device_or_dispatch_performed: bool,
    model_execution_performed: bool,
    hcli_execution_performed: bool,
    tps_or_tg_measurement_performed: bool,
    required_before_ready: Vec<&'static str>,
    claim_boundary: Vec<&'static str>,
    unsealed_preimage_sha256: String,
}

fn is_lower_sha256(value: &str) -> bool {
    value.len() == 64
        && value
            .bytes()
            .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte))
        && value.bytes().any(|byte| byte != b'0')
}

fn evaluate(input: &ReadinessInput) -> ReadinessReport {
    let mut contract_errors = Vec::new();
    let input_schema_valid = input.schema == INPUT_SCHEMA;
    if !input_schema_valid {
        contract_errors.push("input schema drifted".into());
    }
    let source_binding_valid = input.source_binding.validate_exact("input").is_ok();
    if !source_binding_valid {
        contract_errors.push("input source/config/tokenizer/artifact binding drifted".into());
    }
    let tokenizer_errors = input.tokenizer_template_tail_binding.validation_errors();
    let tokenizer_template_tail_binding_valid = tokenizer_errors.is_empty();
    contract_errors.extend(tokenizer_errors);
    let cpu_guard_errors = input.current_cpu_row_tail_guard.validation_errors();
    let current_cpu_row_tail_guard_valid = cpu_guard_errors.is_empty();
    contract_errors.extend(cpu_guard_errors);

    let mut row_assessments = Vec::with_capacity(input.device_row_witnesses.len());
    let mut observed_valid_real_input_kinds = BTreeSet::new();
    let mut seen_receipts = BTreeSet::new();
    let mut session_buffer_owner = BTreeMap::<String, String>::new();
    let mut session_buffer_by_session = BTreeMap::<String, String>::new();
    for (index, witness) in input.device_row_witnesses.iter().enumerate() {
        let failures = witness.validation_errors(index);
        if !seen_receipts.insert(witness.receipt_seal_sha256.clone()) {
            contract_errors.push(format!(
                "device row {index} duplicates a sealed device-row receipt; each real input witness must be unique"
            ));
        }
        let session = witness.session_buffer.session_id_sha256.clone();
        let buffer = witness
            .session_buffer
            .embedding_output_buffer_id_sha256
            .clone();
        if let Some(owner) = session_buffer_owner.insert(buffer.clone(), session.clone()) {
            if owner != session {
                contract_errors.push(format!(
                    "device rows alias output buffer {buffer} across logical sessions {owner} and {session}"
                ));
            }
        }
        if let Some(previous_buffer) =
            session_buffer_by_session.insert(session.clone(), buffer.clone())
        {
            if previous_buffer != buffer {
                contract_errors.push(format!(
                    "logical session {session} changed its owned embedding output buffer without an explicit session-generation bridge"
                ));
            }
        }
        let satisfied = failures.is_empty();
        if satisfied {
            observed_valid_real_input_kinds.insert(witness.input_kind);
        }
        row_assessments.push(RowAssessment {
            index,
            input_kind: witness.input_kind,
            token_id: witness.token_id,
            session_id_sha256: session,
            embedding_output_buffer_id_sha256: buffer,
            satisfied,
            failures,
        });
    }
    let required_input_kinds = required_real_input_kinds();
    if input.device_row_witnesses.len() < required_input_kinds.len() {
        contract_errors.push(
            "embedding device readiness requires at least one real source-template prompt row and one real feedback row"
                .into(),
        );
    }
    for input_kind in &required_input_kinds {
        if !observed_valid_real_input_kinds.contains(input_kind) {
            contract_errors.push(format!(
                "embedding device readiness lacks a valid real {input_kind:?} row parity witness"
            ));
        }
    }

    let tail_refusal_failures = input
        .tail_refusal_witness
        .as_ref()
        .map(TailRefusalDeviceWitness::validation_errors)
        .unwrap_or_else(|| vec!["no sealed device tail-refusal witness was supplied".into()]);
    let tail_refusal_witness_valid = tail_refusal_failures.is_empty();
    let embedding_device_readiness_earned = input_schema_valid
        && source_binding_valid
        && tokenizer_template_tail_binding_valid
        && current_cpu_row_tail_guard_valid
        && row_assessments
            .iter()
            .all(|assessment| assessment.satisfied)
        && observed_valid_real_input_kinds == required_input_kinds
        && tail_refusal_witness_valid
        && contract_errors.is_empty();
    let status = if embedding_device_readiness_earned {
        "READY_FOR_QWEN80_EMBEDDING_DEVICE_INTEGRATION_NOT_A_COMPLETE_DECODER"
    } else {
        "INCOMPLETE_QWEN80_EMBEDDING_DEVICE_READINESS_CPU_ROW_TAIL_GUARD_HAS_NO_FULL_SOURCE_ARTIFACT_BOUND_REAL_INPUT_DEVICE_PARITY"
    };
    let mut report = ReadinessReport {
        schema: RESULT_SCHEMA,
        status,
        embedding_device_readiness_earned,
        complete_decoder_readiness_earned: false,
        real_gravity_server_launch_precondition_satisfied: false,
        input_schema_valid,
        source_binding_valid,
        tokenizer_template_tail_binding_valid,
        current_cpu_row_tail_guard_valid,
        current_cpu_guard_is_device_evidence: false,
        required_real_input_kinds: required_input_kinds.into_iter().collect(),
        observed_valid_real_input_kinds: observed_valid_real_input_kinds.into_iter().collect(),
        row_assessments,
        tail_refusal_witness_valid,
        tail_refusal_failures,
        contract_errors,
        read_only_contract: true,
        live_artifact_scan_performed: false,
        metal_device_or_dispatch_performed: false,
        model_execution_performed: false,
        hcli_execution_performed: false,
        tps_or_tg_measurement_performed: false,
        required_before_ready: vec![
            "Seal source/artifact-bound device gather parity for at least one real source-template prompt ID and one real valid feedback ID, each against the admitted direct-packed CPU row oracle.",
            "For every device row, retain token/template provenance, exact [151936,2048] group128 ABI, a valid source ID <151669, bounded finite [2048] CPU/device parity, and a unique sealed receipt.",
            "Prove a per-session owned [2048] f32 (8192-byte) embedding output buffer is allocated before gather, retained through the layer-0 input fence, and not aliased across logical sessions.",
            "Seal a source/artifact-bound tail refusal witness proving IDs 151669..151935 cannot gather, write an embedding buffer, enter layer 0, or mutate session state.",
            "After this narrow embedding frontier is satisfied, independently satisfy all 48 DeltaNet/GQA/MoE boundaries, state, command graph, terminal head, tokenizer feedback, and clean TPS/TG gates before any decoder/server claim.",
        ],
        claim_boundary: vec![
            "The current CPU row/tail guard is component evidence only. It is not a native device gather, full decoder, token, server, HCLI, TPS, TG, or tournament result.",
            "This ledger validates metadata only: it does not open artifacts, allocate a device buffer, dispatch Metal, execute a model, or create a logical session.",
            "Even READY_FOR_QWEN80_EMBEDDING_DEVICE_INTEGRATION_NOT_A_COMPLETE_DECODER is not permission to expose a Gravity server or report BASE_TRUE_TPS.",
        ],
        unsealed_preimage_sha256: String::new(),
    };
    report.unsealed_preimage_sha256 = format!(
        "{:x}",
        Sha256::digest(serde_json::to_vec(&report).unwrap_or_default())
    );
    report
}

fn current_evidence_input() -> ReadinessInput {
    ReadinessInput {
        schema: INPUT_SCHEMA.into(),
        source_binding: SourceBinding::exact(),
        tokenizer_template_tail_binding: TokenizerTemplateTailBinding::exact(),
        current_cpu_row_tail_guard: CurrentCpuRowTailGuard::exact(),
        device_row_witnesses: Vec::new(),
        tail_refusal_witness: None,
    }
}

fn write_report_atomic(path: &Path, report: &ReadinessReport) -> Result<(), Box<dyn Error>> {
    let parent = path.parent().ok_or("output path has no parent")?;
    if !parent.is_dir() {
        return Err(format!("output parent is missing: {}", parent.display()).into());
    }
    let temporary = path.with_extension(format!("tmp-{}", std::process::id()));
    fs::write(&temporary, serde_json::to_vec_pretty(report)?)?;
    fs::rename(&temporary, path)?;
    Ok(())
}

enum InputMode {
    Inventory(PathBuf),
    CurrentEvidence,
}

struct Arguments {
    input_mode: InputMode,
    out: PathBuf,
}

fn usage() -> &'static str {
    "usage: ascension_qwen80_embedding_device_readiness_contract \\\n+--current-evidence|--input ABSOLUTE_JSON --out ABSOLUTE_JSON"
}

fn parse_args() -> Result<Arguments, Box<dyn Error>> {
    let mut inventory = None;
    let mut current_evidence = false;
    let mut out = None;
    let mut args = env::args().skip(1);
    while let Some(flag) = args.next() {
        match flag.as_str() {
            "--input" => {
                let value = args.next().ok_or("missing absolute path after --input")?;
                if inventory.replace(PathBuf::from(value)).is_some() {
                    return Err("--input supplied more than once".into());
                }
            }
            "--current-evidence" => {
                if current_evidence {
                    return Err("--current-evidence supplied more than once".into());
                }
                current_evidence = true;
            }
            "--out" => {
                let value = args.next().ok_or("missing absolute path after --out")?;
                if out.replace(PathBuf::from(value)).is_some() {
                    return Err("--out supplied more than once".into());
                }
            }
            _ => return Err(format!("unsupported option {flag:?}; {}", usage()).into()),
        }
    }
    if current_evidence == inventory.is_some() {
        return Err(format!(
            "supply exactly one of --current-evidence or --input; {}",
            usage()
        )
        .into());
    }
    let input_mode = if current_evidence {
        InputMode::CurrentEvidence
    } else {
        let path = inventory.expect("checked inventory presence");
        if !path.is_absolute() {
            return Err("--input must be absolute".into());
        }
        InputMode::Inventory(path)
    };
    let out = out.ok_or("missing --out")?;
    if !out.is_absolute() {
        return Err("--out must be absolute".into());
    }
    Ok(Arguments { input_mode, out })
}

fn main() {
    let result = (|| -> Result<(), Box<dyn Error>> {
        let args = parse_args()?;
        let input = match args.input_mode {
            InputMode::CurrentEvidence => current_evidence_input(),
            InputMode::Inventory(path) => serde_json::from_slice(&fs::read(path)?)?,
        };
        let report = evaluate(&input);
        write_report_atomic(&args.out, &report)?;
        if !report.embedding_device_readiness_earned {
            return Err(format!(
                "Qwen80 embedding device readiness is incomplete; report written to {}",
                args.out.display()
            )
            .into());
        }
        Ok(())
    })();
    if let Err(error) = result {
        eprintln!("ascension_qwen80_embedding_device_readiness_contract: {error}");
        std::process::exit(2);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn test_sha(seed: usize) -> String {
        format!("{:064x}", seed + 1)
    }

    fn real_row(
        input_kind: RealInputKind,
        token_id: u32,
        session_seed: usize,
    ) -> DeviceRowParityWitness {
        let device_row_sha256 = test_sha(session_seed * 100 + 1);
        DeviceRowParityWitness {
            schema: DEVICE_ROW_SCHEMA.into(),
            receipt_seal_sha256: test_sha(session_seed * 100 + 2),
            source_binding: SourceBinding::exact(),
            embedding: EmbeddingAbi::exact(),
            embedding_artifact_sha256: test_sha(session_seed * 100 + 3),
            tokenizer_contract_document_sha256: TOKENIZER_CONTRACT_DOCUMENT_SHA256.into(),
            cpu_guard_schema: CPU_GUARD_SCHEMA.into(),
            cpu_guard_status: CPU_GUARD_STATUS.into(),
            cpu_oracle_api: CPU_ORACLE_API.into(),
            input_kind,
            token_id,
            source_input_sha256: test_sha(session_seed * 100 + 4),
            source_tokenizer_encoded: true,
            source_template_or_feedback_chain_bound: true,
            input_is_real_source_token: true,
            synthetic_input: false,
            fixture_only: false,
            fallback_used: false,
            reserved_tail_preflight_passed_before_gather: true,
            backend: "metal".into(),
            device_dispatches: 1,
            source_bound: true,
            artifact_bound: true,
            device_embedding_gather_performed: true,
            cpu_row_sha256: test_sha(session_seed * 100 + 5),
            device_row_sha256: device_row_sha256.clone(),
            max_abs_error: 0.000_001,
            parity_tolerance: 0.000_01,
            session_buffer: SessionEmbeddingBufferProvenance {
                session_id_sha256: test_sha(session_seed * 100 + 6),
                logical_session_generation: 1,
                allocation_receipt_sha256: test_sha(session_seed * 100 + 7),
                token_input_buffer_id_sha256: test_sha(session_seed * 100 + 8),
                embedding_output_buffer_id_sha256: test_sha(session_seed * 100 + 9),
                output_offset_bytes: 0,
                output_capacity_bytes: EMBEDDING_BUFFER_BYTES,
                output_element_count: HIDDEN,
                output_row_sha256: device_row_sha256.clone(),
                per_session_owned: true,
                not_aliased_to_another_logical_session: true,
                allocated_before_gather: true,
                retained_through_layer0_input_fence: true,
                layer0_input_sha256: device_row_sha256,
            },
        }
    }

    fn tail_refusal() -> TailRefusalDeviceWitness {
        TailRefusalDeviceWitness {
            schema: TAIL_REFUSAL_SCHEMA.into(),
            receipt_seal_sha256: test_sha(9_000),
            source_binding: SourceBinding::exact(),
            embedding: EmbeddingAbi::exact(),
            embedding_artifact_sha256: test_sha(9_001),
            first_reserved_id: FIRST_RESERVED_ID,
            last_reserved_id: LAST_RESERVED_ID,
            reserved_tail_rows: RESERVED_TAIL_ROWS,
            backend: "metal".into(),
            source_bound: true,
            artifact_bound: true,
            device_tail_preflight_performed: true,
            first_reserved_rejected_before_gather: true,
            last_reserved_rejected_before_gather: true,
            every_reserved_id_rejected_before_gather: true,
            no_reserved_embedding_buffer_write: true,
            no_reserved_layer0_input: true,
            no_reserved_state_mutation: true,
            fixture_only: false,
            fallback_used: false,
        }
    }

    fn fully_physical_input() -> ReadinessInput {
        let mut input = current_evidence_input();
        input.device_row_witnesses = vec![
            real_row(RealInputKind::SourceTemplatePrompt, 151_643, 1),
            real_row(RealInputKind::ValidAutoregressiveFeedback, 7, 2),
        ];
        input.tail_refusal_witness = Some(tail_refusal());
        input
    }

    #[test]
    fn exact_geometry_tokenizer_tail_and_cpu_guard_are_bound() {
        let input = current_evidence_input();
        assert_eq!(EmbeddingAbi::exact().shape, vec![151_936, 2_048]);
        assert_eq!(EmbeddingAbi::exact().direct_packed_group_size, 128);
        assert_eq!(TOKENIZER_VOCAB, 151_669);
        assert_eq!(RESERVED_TAIL_ROWS, 267);
        assert!(input
            .tokenizer_template_tail_binding
            .validation_errors()
            .is_empty());
        assert!(input
            .current_cpu_row_tail_guard
            .validation_errors()
            .is_empty());
    }

    #[test]
    fn current_cpu_row_tail_guard_remains_incomplete_without_device_evidence() {
        let report = evaluate(&current_evidence_input());
        assert!(!report.embedding_device_readiness_earned);
        assert_eq!(
            report.status,
            "INCOMPLETE_QWEN80_EMBEDDING_DEVICE_READINESS_CPU_ROW_TAIL_GUARD_HAS_NO_FULL_SOURCE_ARTIFACT_BOUND_REAL_INPUT_DEVICE_PARITY"
        );
        assert!(report.current_cpu_row_tail_guard_valid);
        assert!(!report.current_cpu_guard_is_device_evidence);
        assert_eq!(report.row_assessments.len(), 0);
        assert!(!report.tail_refusal_witness_valid);
        assert!(!report.complete_decoder_readiness_earned);
    }

    #[test]
    fn rejects_any_reserved_tail_token_before_device_gather() {
        let mut input = fully_physical_input();
        input.device_row_witnesses[0].token_id = FIRST_RESERVED_ID;
        let report = evaluate(&input);
        assert!(!report.embedding_device_readiness_earned);
        assert!(report.row_assessments[0]
            .failures
            .iter()
            .any(|failure| failure.contains("reserved tail")));
    }

    #[test]
    fn rejects_embedding_geometry_or_source_artifact_drift() {
        let mut input = fully_physical_input();
        input.device_row_witnesses[0].embedding.shape = vec![TOKENIZER_VOCAB as usize, HIDDEN];
        input.device_row_witnesses[1].source_binding.source_revision = "wrong".into();
        let report = evaluate(&input);
        assert!(!report.embedding_device_readiness_earned);
        assert!(report.row_assessments[0]
            .failures
            .iter()
            .any(|failure| failure.contains("embedding ABI")));
        assert!(report.row_assessments[1]
            .failures
            .iter()
            .any(|failure| failure.contains("binding drifted")));
    }

    #[test]
    fn rejects_fixture_fallback_or_missing_real_feedback_coverage() {
        let mut input = fully_physical_input();
        input.device_row_witnesses[1].synthetic_input = true;
        input.device_row_witnesses[1].fallback_used = true;
        let report = evaluate(&input);
        assert!(!report.embedding_device_readiness_earned);
        assert!(report.row_assessments[1]
            .failures
            .iter()
            .any(|failure| failure.contains("not a real tokenizer")));
        assert!(report
            .contract_errors
            .iter()
            .any(|failure| failure.contains("ValidAutoregressiveFeedback")));
    }

    #[test]
    fn rejects_cross_session_embedding_output_buffer_alias() {
        let mut input = fully_physical_input();
        input.device_row_witnesses[1]
            .session_buffer
            .embedding_output_buffer_id_sha256 = input.device_row_witnesses[0]
            .session_buffer
            .embedding_output_buffer_id_sha256
            .clone();
        let report = evaluate(&input);
        assert!(!report.embedding_device_readiness_earned);
        assert!(report
            .contract_errors
            .iter()
            .any(|failure| failure.contains("alias output buffer")));
    }

    #[test]
    fn exact_hypothetical_device_ledger_satisfies_only_embedding_frontier() {
        let report = evaluate(&fully_physical_input());
        assert!(report.embedding_device_readiness_earned);
        assert!(report.tail_refusal_witness_valid);
        assert_eq!(report.observed_valid_real_input_kinds.len(), 2);
        assert!(!report.complete_decoder_readiness_earned);
        assert!(!report.real_gravity_server_launch_precondition_satisfied);
    }
}

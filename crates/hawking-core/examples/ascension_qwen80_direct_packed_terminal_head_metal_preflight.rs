//! Build-only Qwen3-Coder-Next direct-packed terminal-head Metal preflight.
//!
//! This file and its paired shader are intentionally unregistered.  They do
//! not import a Metal context, compile a library, open an artifact, or dispatch
//! a command buffer.  Instead, this is a fail-closed ABI/capture ledger for a
//! future strictly leased terminal component:
//!
//! `real post-48-layer hidden [2048] -> final RMSNorm -> all 151936 lm-head
//! rows -> mask 151669..151935 -> deterministic lowest-ID-tie sample ->
//! tokenizer-addressable feedback`.
//!
//! It must remain a preflight until a separate source/admission-bound, durable
//! capture supplies a real all-layer hidden device buffer, a sealed CPU
//! baseline, a fresh non-timed lease, and a receipt-last outer capture.

use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::env;
use std::error::Error;
use std::fs;
use std::path::{Path, PathBuf};

const INPUT_SCHEMA: &str =
    "hawking.ascension.qwen80_direct_packed_terminal_head_metal_preflight_input.v1";
const RESULT_SCHEMA: &str =
    "hawking.ascension.qwen80_direct_packed_terminal_head_metal_preflight_result.v1";
const CAPTURE_SCHEMA: &str =
    "hawking.ascension.qwen80_direct_packed_terminal_head_metal_capture.v1";
const CAPTURE_STATUS: &str =
    "CAPTURED_QWEN80_DIRECT_PACKED_TERMINAL_HEAD_STRICT_MATH_COMPONENT_ONLY";
const LEASE_SCHEMA: &str =
    "hawking.ascension.qwen80_direct_packed_terminal_head_metal_quiet_lease.v1";
const LEASE_STATUS: &str =
    "GRANTED_QWEN80_DIRECT_PACKED_TERMINAL_HEAD_STRICT_MATH_NON_TIMED_COMPONENT_LEASE";
const HIDDEN_SCHEMA: &str = "hawking.ascension.qwen80_post48_layer_hidden_device_buffer.v1";
const BASELINE_SCHEMA: &str = "hawking.ascension.qwen80_terminal_head_cpu_baseline_wrapper.v1";
const BASELINE_STATUS: &str =
    "SEALED_CURRENT_ADMITTED_QWEN80_TERMINAL_HEAD_AND_SAMPLER_CPU_BASELINE";
const TERMINAL_RECEIPT_SCHEMA: &str = "hawking.ascension.qwen80_direct_packed_terminal_head_cpu.v1";
const TERMINAL_RECEIPT_STATUS: &str =
    "EARNED_CURRENT_ADMITTED_QWEN80_DIRECT_PACKED_TERMINAL_COMPONENT_CPU_ONLY_NOT_RUNTIME_OR_TOKEN";
const TERMINAL_RECEIPT_DOCUMENT_SHA256: &str =
    "1ebe19139833491ec06cc7515f6844fad0a122de15fb74c978dfda3524a38d04";
const TERMINAL_RECEIPT_UNSEALED_PREIMAGE_SHA256: &str =
    "d815c6bfff615a1c238ed56863b14ba61349f1c04824195448722a6a3e81372b";
const TOKENIZER_RECEIPT_SCHEMA: &str =
    "hawking.ascension.qwen80_tokenizer_sampler_handoff_contract.v1";
const TOKENIZER_RECEIPT_STATUS: &str =
    "EARNED_SOURCE_BOUND_TOKENIZER_TEMPLATE_SAMPLER_HANDOFF_COMPONENT_NOT_RUNTIME_OR_TOKEN";
const TOKENIZER_RECEIPT_DOCUMENT_SHA256: &str =
    "e152b21d9eae43e7039f9d646412b2806b8f07d3d3c7ea932ab281dc6c9a0792";
const TOKENIZER_RECEIPT_UNSEALED_PREIMAGE_SHA256: &str =
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
const FINAL_NORM_NAME: &str = "model.norm.weight";
const LM_HEAD_NAME: &str = "lm_head.weight";
const FINAL_NORM_ARTIFACT_SHA256: &str =
    "6306499804d27e48f0a041e94d366feae5cbf8436fac15815a559a15717ef36e";
const LM_HEAD_ARTIFACT_SHA256: &str =
    "549c448be683ed00ec792329c5167f3f0cacfcb3af339a1fb064ed0a004d9998";
const PACKED_MAGIC: &str = "HQ30G1B1";
const PACKED_VERSION: usize = 1;
const HIDDEN: usize = 2_048;
const LM_HEAD_ROWS: usize = 151_936;
const TOKENIZER_VOCAB: u32 = 151_669;
const FIRST_RESERVED_ID: u32 = TOKENIZER_VOCAB;
const LAST_RESERVED_ID: u32 = LM_HEAD_ROWS as u32 - 1;
const RESERVED_TAIL_ROWS: u32 = LM_HEAD_ROWS as u32 - TOKENIZER_VOCAB;
const GROUP_SIZE: usize = 128;
const RMS_EPSILON_BITS: u32 = 897_988_541;
const POST48_HIDDEN_BYTES: usize = HIDDEN * std::mem::size_of::<f32>();
const DETERMINISTIC_SAMPLER: &str = "greedy_argmax_lowest_token_id_tie_break";
const STAGED_SHADER: &str =
    include_str!("../shaders/qwen80_direct_packed_terminal_head_preflight.metal");

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
struct SourceAdmissionBinding {
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
    rms_norm_epsilon_bits: u32,
}

impl SourceAdmissionBinding {
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
            rms_norm_epsilon_bits: RMS_EPSILON_BITS,
        }
    }

    fn validate_exact(&self, label: &str) -> Result<(), String> {
        if self != &Self::exact() {
            return Err(format!(
                "{label} source/admission/tokenizer binding drifted"
            ));
        }
        Ok(())
    }
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
struct PackedTensorAbi {
    name: String,
    shape: Vec<usize>,
    group_size: usize,
    magic: String,
    version: usize,
    artifact_sha256: String,
}

impl PackedTensorAbi {
    fn final_norm() -> Self {
        Self {
            name: FINAL_NORM_NAME.into(),
            shape: vec![HIDDEN],
            group_size: GROUP_SIZE,
            magic: PACKED_MAGIC.into(),
            version: PACKED_VERSION,
            artifact_sha256: FINAL_NORM_ARTIFACT_SHA256.into(),
        }
    }

    fn lm_head() -> Self {
        Self {
            name: LM_HEAD_NAME.into(),
            shape: vec![LM_HEAD_ROWS, HIDDEN],
            group_size: GROUP_SIZE,
            magic: PACKED_MAGIC.into(),
            version: PACKED_VERSION,
            artifact_sha256: LM_HEAD_ARTIFACT_SHA256.into(),
        }
    }

    fn validate_final_norm(&self, label: &str) -> Result<(), String> {
        if self != &Self::final_norm() {
            return Err(format!(
                "{label} final norm ABI must be direct-packed model.norm.weight [2048], group128"
            ));
        }
        Ok(())
    }

    fn validate_lm_head(&self, label: &str) -> Result<(), String> {
        if self != &Self::lm_head() {
            return Err(format!(
                "{label} lm_head ABI must be direct-packed lm_head.weight [151936,2048], group128"
            ));
        }
        Ok(())
    }
}

#[derive(Clone, Debug, Deserialize, Serialize)]
struct SealedCpuBaseline {
    schema: String,
    status: String,
    seal_sha256: String,
    integrity_verified: bool,
    source_admission: SourceAdmissionBinding,
    terminal_receipt_schema: String,
    terminal_receipt_status: String,
    terminal_receipt_document_sha256: String,
    terminal_receipt_unsealed_preimage_sha256: String,
    tokenizer_receipt_schema: String,
    tokenizer_receipt_status: String,
    tokenizer_receipt_document_sha256: String,
    tokenizer_receipt_unsealed_preimage_sha256: String,
}

impl SealedCpuBaseline {
    fn validation_errors(&self) -> Vec<String> {
        let mut errors = Vec::new();
        if self.schema != BASELINE_SCHEMA
            || self.status != BASELINE_STATUS
            || !is_lower_sha256(&self.seal_sha256)
            || !self.integrity_verified
        {
            errors.push("terminal CPU baseline must be sealed and integrity-verified".into());
        }
        if let Err(error) = self
            .source_admission
            .validate_exact("terminal CPU baseline")
        {
            errors.push(error);
        }
        if self.terminal_receipt_schema != TERMINAL_RECEIPT_SCHEMA
            || self.terminal_receipt_status != TERMINAL_RECEIPT_STATUS
            || self.terminal_receipt_document_sha256 != TERMINAL_RECEIPT_DOCUMENT_SHA256
            || self.terminal_receipt_unsealed_preimage_sha256
                != TERMINAL_RECEIPT_UNSEALED_PREIMAGE_SHA256
            || self.tokenizer_receipt_schema != TOKENIZER_RECEIPT_SCHEMA
            || self.tokenizer_receipt_status != TOKENIZER_RECEIPT_STATUS
            || self.tokenizer_receipt_document_sha256 != TOKENIZER_RECEIPT_DOCUMENT_SHA256
            || self.tokenizer_receipt_unsealed_preimage_sha256
                != TOKENIZER_RECEIPT_UNSEALED_PREIMAGE_SHA256
        {
            errors.push(
                "terminal CPU baseline does not bind the exact current terminal/tokenizer component receipts"
                    .into(),
            );
        }
        errors
    }
}

#[derive(Clone, Debug, Deserialize, Serialize)]
struct Post48LayerHiddenBuffer {
    schema: String,
    source_admission: SourceAdmissionBinding,
    buffer_id_sha256: String,
    command_graph_capture_id_sha256: String,
    all_layer_hidden_sha256: String,
    device_parity_receipt_seal_sha256: String,
    shape: Vec<usize>,
    byte_length: usize,
    produced_by_exact_48_layer_schedule: bool,
    all_48_layers_physically_completed: bool,
    source_token_or_feedback_provenance_sha256: String,
    synthetic_or_component_fixture: bool,
    fallback_used: bool,
    buffer_owned_by_logical_session: bool,
    retained_until_terminal_feedback_fence: bool,
}

impl Post48LayerHiddenBuffer {
    fn validation_errors(&self) -> Vec<String> {
        let mut errors = Vec::new();
        if self.schema != HIDDEN_SCHEMA {
            errors.push("post-48 hidden buffer schema drifted".into());
        }
        if let Err(error) = self
            .source_admission
            .validate_exact("post-48 hidden buffer")
        {
            errors.push(error);
        }
        for (label, digest) in [
            ("buffer", self.buffer_id_sha256.as_str()),
            (
                "command_graph_capture",
                self.command_graph_capture_id_sha256.as_str(),
            ),
            ("all_layer_hidden", self.all_layer_hidden_sha256.as_str()),
            (
                "device_parity_receipt",
                self.device_parity_receipt_seal_sha256.as_str(),
            ),
            (
                "source_token_or_feedback_provenance",
                self.source_token_or_feedback_provenance_sha256.as_str(),
            ),
        ] {
            if !is_lower_sha256(digest) {
                errors.push(format!("post-48 hidden buffer has invalid {label} digest"));
            }
        }
        if self.shape.as_slice() != [HIDDEN]
            || self.byte_length != POST48_HIDDEN_BYTES
            || !self.produced_by_exact_48_layer_schedule
            || !self.all_48_layers_physically_completed
            || self.synthetic_or_component_fixture
            || self.fallback_used
            || !self.buffer_owned_by_logical_session
            || !self.retained_until_terminal_feedback_fence
        {
            errors.push(
                "terminal preflight requires a real per-session post-48-layer [2048] f32 (8192-byte) hidden buffer with no fixture/fallback"
                    .into(),
            );
        }
        errors
    }
}

#[derive(Clone, Copy, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(rename_all = "snake_case")]
enum TerminalStage {
    BindRealPost48LayerHidden,
    FinalRmsNorm,
    AllRowLmHead,
    MaskReservedTail,
    DeterministicSample,
    ValidateFeedback,
}

fn expected_terminal_order() -> Vec<TerminalStage> {
    vec![
        TerminalStage::BindRealPost48LayerHidden,
        TerminalStage::FinalRmsNorm,
        TerminalStage::AllRowLmHead,
        TerminalStage::MaskReservedTail,
        TerminalStage::DeterministicSample,
        TerminalStage::ValidateFeedback,
    ]
}

#[derive(Clone, Debug, Deserialize, Serialize)]
struct TerminalHeadCaptureLease {
    schema: String,
    status: String,
    lease_id_sha256: String,
    lease_seal_sha256: String,
    source_admission: SourceAdmissionBinding,
    baseline_seal_sha256: String,
    command_graph_capture_id_sha256: String,
    expected_post48_hidden_sha256: String,
    fresh_for_this_exact_capture: bool,
    strict_math_required: bool,
    timing_or_benchmarking_allowed: bool,
    complete_decoder_or_token_claim_allowed: bool,
    tps_or_tg_claim_allowed: bool,
    cpu_or_bf16_fallback_allowed: bool,
    selected_row_lm_head_allowed: bool,
    automatic_retry_prohibited: bool,
    outer_reaped_capture_required: bool,
    terminal_receipt_written_last: bool,
}

impl TerminalHeadCaptureLease {
    fn validation_errors(
        &self,
        baseline: &SealedCpuBaseline,
        hidden: &Post48LayerHiddenBuffer,
    ) -> Vec<String> {
        let mut errors = Vec::new();
        if self.schema != LEASE_SCHEMA
            || self.status != LEASE_STATUS
            || !is_lower_sha256(&self.lease_id_sha256)
            || !is_lower_sha256(&self.lease_seal_sha256)
        {
            errors.push("terminal-head lease schema/status/seal drifted".into());
        }
        if let Err(error) = self.source_admission.validate_exact("terminal-head lease") {
            errors.push(error);
        }
        if self.baseline_seal_sha256 != baseline.seal_sha256
            || self.command_graph_capture_id_sha256 != hidden.command_graph_capture_id_sha256
            || self.expected_post48_hidden_sha256 != hidden.all_layer_hidden_sha256
        {
            errors.push(
                "terminal-head lease does not bind the sealed baseline and exact post-48 hidden capture"
                    .into(),
            );
        }
        if !self.fresh_for_this_exact_capture
            || !self.strict_math_required
            || self.timing_or_benchmarking_allowed
            || self.complete_decoder_or_token_claim_allowed
            || self.tps_or_tg_claim_allowed
            || self.cpu_or_bf16_fallback_allowed
            || self.selected_row_lm_head_allowed
            || !self.automatic_retry_prohibited
            || !self.outer_reaped_capture_required
            || !self.terminal_receipt_written_last
        {
            errors.push(
                "terminal-head lease must be fresh, strict, receipt-last, outer-reaped, non-timed, and component-only"
                    .into(),
            );
        }
        errors
    }
}

#[derive(Clone, Debug, Deserialize, Serialize)]
struct FutureTerminalHeadCapture {
    schema: String,
    status: String,
    receipt_seal_sha256: String,
    source_admission: SourceAdmissionBinding,
    baseline_seal_sha256: String,
    lease_seal_sha256: String,
    command_graph_capture_id_sha256: String,
    post48_hidden_buffer_id_sha256: String,
    post48_hidden_sha256: String,
    ordered_stages: Vec<TerminalStage>,
    final_norm: PackedTensorAbi,
    lm_head: PackedTensorAbi,
    backend: String,
    device_dispatches: usize,
    actual_device_execution: bool,
    all_lm_head_rows_evaluated: usize,
    raw_logits_sha256: String,
    first_reserved_id: u32,
    last_reserved_id: u32,
    reserved_tail_rows: u32,
    every_reserved_logit_negative_infinity: bool,
    sampler_policy: String,
    sampled_token_id: u32,
    sampled_token_is_tokenizer_addressable: bool,
    feedback_token_id: u32,
    feedback_matches_sampled_token: bool,
    feedback_validated_before_next_embedding_or_state_step: bool,
    final_fence_before_capture_receipt: bool,
    fixture_only: bool,
    fallback_used: bool,
    selected_row_shortcut_used: bool,
}

impl FutureTerminalHeadCapture {
    fn validation_errors(
        &self,
        baseline: &SealedCpuBaseline,
        hidden: &Post48LayerHiddenBuffer,
        lease: &TerminalHeadCaptureLease,
    ) -> Vec<String> {
        let mut errors = Vec::new();
        if self.schema != CAPTURE_SCHEMA
            || self.status != CAPTURE_STATUS
            || !is_lower_sha256(&self.receipt_seal_sha256)
        {
            errors.push("future terminal capture schema/status/seal drifted".into());
        }
        if let Err(error) = self
            .source_admission
            .validate_exact("future terminal capture")
        {
            errors.push(error);
        }
        if self.baseline_seal_sha256 != baseline.seal_sha256
            || self.lease_seal_sha256 != lease.lease_seal_sha256
            || self.command_graph_capture_id_sha256 != hidden.command_graph_capture_id_sha256
            || self.post48_hidden_buffer_id_sha256 != hidden.buffer_id_sha256
            || self.post48_hidden_sha256 != hidden.all_layer_hidden_sha256
        {
            errors.push(
                "future terminal capture does not join the exact baseline/lease/post-48 hidden buffer"
                    .into(),
            );
        }
        if self.ordered_stages != expected_terminal_order() {
            errors.push(
                "terminal capture order must be post-48 hidden -> RMSNorm -> all 151936 head rows -> tail mask -> deterministic sample -> feedback"
                    .into(),
            );
        }
        if let Err(error) = self
            .final_norm
            .validate_final_norm("future terminal capture")
        {
            errors.push(error);
        }
        if let Err(error) = self.lm_head.validate_lm_head("future terminal capture") {
            errors.push(error);
        }
        if self.backend != "metal"
            || self.device_dispatches < 5
            || !self.actual_device_execution
            || self.all_lm_head_rows_evaluated != LM_HEAD_ROWS
            || !is_lower_sha256(&self.raw_logits_sha256)
            || self.fixture_only
            || self.fallback_used
            || self.selected_row_shortcut_used
        {
            errors.push(
                "future terminal capture requires real strict device execution of all 151936 direct-packed head rows without fixture/fallback/selected-row shortcut"
                    .into(),
            );
        }
        if self.first_reserved_id != FIRST_RESERVED_ID
            || self.last_reserved_id != LAST_RESERVED_ID
            || self.reserved_tail_rows != RESERVED_TAIL_ROWS
            || !self.every_reserved_logit_negative_infinity
        {
            errors.push(
                "terminal capture tail mask must set exactly IDs 151669..151935 to -infinity"
                    .into(),
            );
        }
        if self.sampler_policy != DETERMINISTIC_SAMPLER
            || self.sampled_token_id >= TOKENIZER_VOCAB
            || !self.sampled_token_is_tokenizer_addressable
            || self.feedback_token_id >= TOKENIZER_VOCAB
            || !self.feedback_matches_sampled_token
            || !self.feedback_validated_before_next_embedding_or_state_step
            || !self.final_fence_before_capture_receipt
        {
            errors.push(
                "terminal capture must deterministically sample only after tail masking and validate an addressable feedback token before next embedding/state"
                    .into(),
            );
        }
        errors
    }
}

#[derive(Clone, Debug, Deserialize, Serialize)]
struct PreflightInput {
    schema: String,
    source_admission: SourceAdmissionBinding,
    sealed_cpu_baseline: Option<SealedCpuBaseline>,
    post48_hidden_buffer: Option<Post48LayerHiddenBuffer>,
    capture_lease: Option<TerminalHeadCaptureLease>,
    terminal_capture: Option<FutureTerminalHeadCapture>,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize)]
struct DispatchSpec {
    kernel: &'static str,
    grid: (u32, u32, u32),
    threadgroup: (u32, u32, u32),
    purpose: &'static str,
}

fn staged_dispatch_plan() -> Vec<DispatchSpec> {
    vec![
        DispatchSpec {
            kernel: "qwen80_terminal_head_final_rmsnorm_direct_packed",
            grid: (256, 1, 1),
            threadgroup: (256, 1, 1),
            purpose: "direct-packed model.norm.weight RMSNorm of real post-48 hidden [2048]",
        },
        DispatchSpec {
            kernel: "qwen80_terminal_head_all_row_direct_packed",
            grid: (256, LM_HEAD_ROWS as u32, 1),
            threadgroup: (256, 1, 1),
            purpose: "every direct-packed lm_head.weight row [151936,2048], never selected rows",
        },
        DispatchSpec {
            kernel: "qwen80_terminal_head_mask_reserved_tail",
            grid: (256, RESERVED_TAIL_ROWS, 1),
            threadgroup: (256, 1, 1),
            purpose: "mask logits IDs 151669 through 151935 to negative infinity",
        },
        DispatchSpec {
            kernel: "qwen80_terminal_head_greedy_sample_lowest_id",
            grid: (1, 1, 1),
            threadgroup: (1, 1, 1),
            purpose: "deterministic greedy argmax with lowest-token-ID tie break after tail mask",
        },
        DispatchSpec {
            kernel: "qwen80_terminal_head_feedback_guard",
            grid: (1, 1, 1),
            threadgroup: (1, 1, 1),
            purpose: "reject non-token feedback before next embedding/state step",
        },
    ]
}

fn validate_staged_shader_and_plan() -> Result<(), String> {
    let expected_kernels = [
        "qwen80_terminal_head_final_rmsnorm_direct_packed",
        "qwen80_terminal_head_all_row_direct_packed",
        "qwen80_terminal_head_mask_reserved_tail",
        "qwen80_terminal_head_greedy_sample_lowest_id",
        "qwen80_terminal_head_feedback_guard",
    ];
    let plan = staged_dispatch_plan();
    if plan.len() != expected_kernels.len()
        || plan.iter().map(|stage| stage.kernel).ne(expected_kernels)
    {
        return Err("terminal-head staged dispatch order drifted".into());
    }
    if plan[0].grid != (256, 1, 1)
        || plan[1].grid != (256, LM_HEAD_ROWS as u32, 1)
        || plan[2].grid != (256, RESERVED_TAIL_ROWS, 1)
        || plan[1].threadgroup != (256, 1, 1)
    {
        return Err("terminal-head staged direct-packed dispatch geometry drifted".into());
    }
    for kernel in expected_kernels {
        if !STAGED_SHADER.contains(&format!("kernel void {kernel}")) {
            return Err(format!("unregistered terminal-head shader lacks {kernel}"));
        }
    }
    for requirement in [
        "qwen80_terminal_head_hidden = 2048u",
        "qwen80_terminal_head_rows = 151936u",
        "qwen80_terminal_head_tokenizer_vocab = 151669u",
        "qwen80_terminal_head_group = 128u",
        "selected_token = candidate",
    ] {
        if !STAGED_SHADER.contains(requirement) {
            return Err(format!(
                "unregistered terminal-head shader lost {requirement:?}"
            ));
        }
    }
    Ok(())
}

#[derive(Serialize)]
struct PreflightReport {
    schema: &'static str,
    status: &'static str,
    terminal_head_preflight_ready_for_separate_device_lease: bool,
    complete_decoder_readiness_earned: bool,
    real_gravity_server_launch_precondition_satisfied: bool,
    input_schema_valid: bool,
    source_admission_valid: bool,
    staged_shader_and_plan_valid: bool,
    sealed_cpu_baseline_valid: bool,
    post48_hidden_buffer_valid: bool,
    capture_lease_valid: bool,
    future_terminal_capture_valid: bool,
    staged_shader_sha256: String,
    staged_dispatch_plan: Vec<DispatchSpec>,
    contract_errors: Vec<String>,
    read_only_build_preflight: bool,
    shader_registered_in_metal_mod: bool,
    live_artifact_scan_performed: bool,
    metal_context_or_dispatch_performed: bool,
    model_execution_performed: bool,
    runtime_watcher_server_started: bool,
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

fn evaluate(input: &PreflightInput) -> PreflightReport {
    let mut contract_errors = Vec::new();
    let input_schema_valid = input.schema == INPUT_SCHEMA;
    if !input_schema_valid {
        contract_errors.push("input schema drifted".into());
    }
    let source_admission_valid = input.source_admission.validate_exact("input").is_ok();
    if !source_admission_valid {
        contract_errors.push("input source/admission binding drifted".into());
    }
    let staged_shader_and_plan_valid = match validate_staged_shader_and_plan() {
        Ok(()) => true,
        Err(error) => {
            contract_errors.push(error);
            false
        }
    };

    let baseline_errors = input
        .sealed_cpu_baseline
        .as_ref()
        .map(SealedCpuBaseline::validation_errors)
        .unwrap_or_else(|| vec!["no sealed terminal CPU baseline supplied".into()]);
    let sealed_cpu_baseline_valid = baseline_errors.is_empty();
    contract_errors.extend(baseline_errors);
    let hidden_errors = input
        .post48_hidden_buffer
        .as_ref()
        .map(Post48LayerHiddenBuffer::validation_errors)
        .unwrap_or_else(|| vec!["no real post-48-layer hidden device buffer supplied".into()]);
    let post48_hidden_buffer_valid = hidden_errors.is_empty();
    contract_errors.extend(hidden_errors);

    let capture_lease_errors = match (&input.sealed_cpu_baseline, &input.post48_hidden_buffer, &input.capture_lease) {
        (Some(baseline), Some(hidden), Some(lease)) => lease.validation_errors(baseline, hidden),
        _ => vec!["cannot validate terminal capture lease before sealed baseline and post-48 hidden buffer exist".into()],
    };
    let capture_lease_valid = capture_lease_errors.is_empty();
    contract_errors.extend(capture_lease_errors);

    let terminal_capture_errors = match (
        &input.sealed_cpu_baseline,
        &input.post48_hidden_buffer,
        &input.capture_lease,
        &input.terminal_capture,
    ) {
        (Some(baseline), Some(hidden), Some(lease), Some(capture)) => {
            capture.validation_errors(baseline, hidden, lease)
        }
        _ => vec!["no future terminal-head component capture joins baseline, lease, and real post-48 hidden buffer".into()],
    };
    let future_terminal_capture_valid = terminal_capture_errors.is_empty();
    contract_errors.extend(terminal_capture_errors);

    let terminal_head_preflight_ready_for_separate_device_lease = input_schema_valid
        && source_admission_valid
        && staged_shader_and_plan_valid
        && sealed_cpu_baseline_valid
        && post48_hidden_buffer_valid
        && capture_lease_valid
        && future_terminal_capture_valid
        && contract_errors.is_empty();
    let status = if terminal_head_preflight_ready_for_separate_device_lease {
        "READY_FOR_QWEN80_TERMINAL_HEAD_SEPARATE_DEVICE_INTEGRATION_NOT_A_COMPLETE_DECODER"
    } else {
        "INCOMPLETE_QWEN80_TERMINAL_HEAD_UNREGISTERED_METAL_PREFLIGHT_REQUIRES_REAL_POST48_HIDDEN_AND_DURABLE_LEASE_CAPTURE"
    };
    let mut report = PreflightReport {
        schema: RESULT_SCHEMA,
        status,
        terminal_head_preflight_ready_for_separate_device_lease,
        complete_decoder_readiness_earned: false,
        real_gravity_server_launch_precondition_satisfied: false,
        input_schema_valid,
        source_admission_valid,
        staged_shader_and_plan_valid,
        sealed_cpu_baseline_valid,
        post48_hidden_buffer_valid,
        capture_lease_valid,
        future_terminal_capture_valid,
        staged_shader_sha256: format!("{:x}", Sha256::digest(STAGED_SHADER.as_bytes())),
        staged_dispatch_plan: staged_dispatch_plan(),
        contract_errors,
        read_only_build_preflight: true,
        shader_registered_in_metal_mod: false,
        live_artifact_scan_performed: false,
        metal_context_or_dispatch_performed: false,
        model_execution_performed: false,
        runtime_watcher_server_started: false,
        hcli_execution_performed: false,
        tps_or_tg_measurement_performed: false,
        required_before_ready: vec![
            "Supply one real source/admission-bound post-48-layer [2048] f32 hidden buffer with a sealed device-parity receipt, exact capture ID, logical-session ownership, and no synthetic/fallback path.",
            "Seal the exact current terminal-head/tokenizer CPU component receipts into the terminal baseline wrapper before compiling or dispatching the staged shader.",
            "Grant one fresh strict non-timed terminal-head lease bound to that baseline and hidden capture; require automatic-retry prohibition, outer reaping, and receipt-last durability.",
            "Capture direct-packed final RMSNorm, every lm_head row 0..151935, exact 267-row tail mask, deterministic lowest-ID-tie sampling, and valid feedback in one fenced command graph.",
            "Only after this narrow component frontier, independently earn 48-layer decoder, state, tokenizer, HCLI, and clean TPS/TG readiness. This preflight never grants them.",
        ],
        claim_boundary: vec![
            "This source is unregistered: it does not modify metal/mod.rs or create a Metal context, compile a library, dispatch a kernel, or scan an artifact.",
            "A future terminal component capture is still not a complete decoder, generated token, Gravity server, HCLI, BASE_TRUE_TPS, TG, capability, Agent OS, or tournament result.",
            "The staged all-row head specifically forbids selected-row shortcuts, CPU/BF16 fallback, timing, and any feedback ID >=151669.",
        ],
        unsealed_preimage_sha256: String::new(),
    };
    report.unsealed_preimage_sha256 = format!(
        "{:x}",
        Sha256::digest(serde_json::to_vec(&report).unwrap_or_default())
    );
    report
}

fn current_evidence_input() -> PreflightInput {
    PreflightInput {
        schema: INPUT_SCHEMA.into(),
        source_admission: SourceAdmissionBinding::exact(),
        sealed_cpu_baseline: None,
        post48_hidden_buffer: None,
        capture_lease: None,
        terminal_capture: None,
    }
}

fn write_report_atomic(path: &Path, report: &PreflightReport) -> Result<(), Box<dyn Error>> {
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
    "usage: ascension_qwen80_direct_packed_terminal_head_metal_preflight \\\n+--current-evidence|--input ABSOLUTE_JSON --out ABSOLUTE_JSON"
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
        if !report.terminal_head_preflight_ready_for_separate_device_lease {
            return Err(format!(
                "Qwen80 terminal-head Metal preflight is incomplete; report written to {}",
                args.out.display()
            )
            .into());
        }
        Ok(())
    })();
    if let Err(error) = result {
        eprintln!("ascension_qwen80_direct_packed_terminal_head_metal_preflight: {error}");
        std::process::exit(2);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn test_sha(seed: usize) -> String {
        format!("{:064x}", seed + 1)
    }

    fn sealed_baseline() -> SealedCpuBaseline {
        SealedCpuBaseline {
            schema: BASELINE_SCHEMA.into(),
            status: BASELINE_STATUS.into(),
            seal_sha256: test_sha(1),
            integrity_verified: true,
            source_admission: SourceAdmissionBinding::exact(),
            terminal_receipt_schema: TERMINAL_RECEIPT_SCHEMA.into(),
            terminal_receipt_status: TERMINAL_RECEIPT_STATUS.into(),
            terminal_receipt_document_sha256: TERMINAL_RECEIPT_DOCUMENT_SHA256.into(),
            terminal_receipt_unsealed_preimage_sha256: TERMINAL_RECEIPT_UNSEALED_PREIMAGE_SHA256
                .into(),
            tokenizer_receipt_schema: TOKENIZER_RECEIPT_SCHEMA.into(),
            tokenizer_receipt_status: TOKENIZER_RECEIPT_STATUS.into(),
            tokenizer_receipt_document_sha256: TOKENIZER_RECEIPT_DOCUMENT_SHA256.into(),
            tokenizer_receipt_unsealed_preimage_sha256: TOKENIZER_RECEIPT_UNSEALED_PREIMAGE_SHA256
                .into(),
        }
    }

    fn post48_hidden() -> Post48LayerHiddenBuffer {
        Post48LayerHiddenBuffer {
            schema: HIDDEN_SCHEMA.into(),
            source_admission: SourceAdmissionBinding::exact(),
            buffer_id_sha256: test_sha(10),
            command_graph_capture_id_sha256: test_sha(11),
            all_layer_hidden_sha256: test_sha(12),
            device_parity_receipt_seal_sha256: test_sha(13),
            shape: vec![HIDDEN],
            byte_length: POST48_HIDDEN_BYTES,
            produced_by_exact_48_layer_schedule: true,
            all_48_layers_physically_completed: true,
            source_token_or_feedback_provenance_sha256: test_sha(14),
            synthetic_or_component_fixture: false,
            fallback_used: false,
            buffer_owned_by_logical_session: true,
            retained_until_terminal_feedback_fence: true,
        }
    }

    fn lease(
        baseline: &SealedCpuBaseline,
        hidden: &Post48LayerHiddenBuffer,
    ) -> TerminalHeadCaptureLease {
        TerminalHeadCaptureLease {
            schema: LEASE_SCHEMA.into(),
            status: LEASE_STATUS.into(),
            lease_id_sha256: test_sha(20),
            lease_seal_sha256: test_sha(21),
            source_admission: SourceAdmissionBinding::exact(),
            baseline_seal_sha256: baseline.seal_sha256.clone(),
            command_graph_capture_id_sha256: hidden.command_graph_capture_id_sha256.clone(),
            expected_post48_hidden_sha256: hidden.all_layer_hidden_sha256.clone(),
            fresh_for_this_exact_capture: true,
            strict_math_required: true,
            timing_or_benchmarking_allowed: false,
            complete_decoder_or_token_claim_allowed: false,
            tps_or_tg_claim_allowed: false,
            cpu_or_bf16_fallback_allowed: false,
            selected_row_lm_head_allowed: false,
            automatic_retry_prohibited: true,
            outer_reaped_capture_required: true,
            terminal_receipt_written_last: true,
        }
    }

    fn capture(
        baseline: &SealedCpuBaseline,
        hidden: &Post48LayerHiddenBuffer,
        lease: &TerminalHeadCaptureLease,
    ) -> FutureTerminalHeadCapture {
        FutureTerminalHeadCapture {
            schema: CAPTURE_SCHEMA.into(),
            status: CAPTURE_STATUS.into(),
            receipt_seal_sha256: test_sha(30),
            source_admission: SourceAdmissionBinding::exact(),
            baseline_seal_sha256: baseline.seal_sha256.clone(),
            lease_seal_sha256: lease.lease_seal_sha256.clone(),
            command_graph_capture_id_sha256: hidden.command_graph_capture_id_sha256.clone(),
            post48_hidden_buffer_id_sha256: hidden.buffer_id_sha256.clone(),
            post48_hidden_sha256: hidden.all_layer_hidden_sha256.clone(),
            ordered_stages: expected_terminal_order(),
            final_norm: PackedTensorAbi::final_norm(),
            lm_head: PackedTensorAbi::lm_head(),
            backend: "metal".into(),
            device_dispatches: 5,
            actual_device_execution: true,
            all_lm_head_rows_evaluated: LM_HEAD_ROWS,
            raw_logits_sha256: test_sha(31),
            first_reserved_id: FIRST_RESERVED_ID,
            last_reserved_id: LAST_RESERVED_ID,
            reserved_tail_rows: RESERVED_TAIL_ROWS,
            every_reserved_logit_negative_infinity: true,
            sampler_policy: DETERMINISTIC_SAMPLER.into(),
            sampled_token_id: TOKENIZER_VOCAB - 1,
            sampled_token_is_tokenizer_addressable: true,
            feedback_token_id: TOKENIZER_VOCAB - 1,
            feedback_matches_sampled_token: true,
            feedback_validated_before_next_embedding_or_state_step: true,
            final_fence_before_capture_receipt: true,
            fixture_only: false,
            fallback_used: false,
            selected_row_shortcut_used: false,
        }
    }

    fn full_future_input() -> PreflightInput {
        let baseline = sealed_baseline();
        let hidden = post48_hidden();
        let lease = lease(&baseline, &hidden);
        let capture = capture(&baseline, &hidden, &lease);
        PreflightInput {
            schema: INPUT_SCHEMA.into(),
            source_admission: SourceAdmissionBinding::exact(),
            sealed_cpu_baseline: Some(baseline),
            post48_hidden_buffer: Some(hidden),
            capture_lease: Some(lease),
            terminal_capture: Some(capture),
        }
    }

    #[test]
    fn staged_shader_and_dispatch_order_bind_exact_terminal_abi() {
        validate_staged_shader_and_plan().unwrap();
        let plan = staged_dispatch_plan();
        assert_eq!(plan.len(), 5);
        assert_eq!(plan[1].grid, (256, 151_936, 1));
        assert_eq!(plan[2].grid, (256, 267, 1));
        assert_eq!(PackedTensorAbi::final_norm().shape, vec![2_048]);
        assert_eq!(PackedTensorAbi::lm_head().shape, vec![151_936, 2_048]);
    }

    #[test]
    fn current_evidence_is_incomplete_and_no_device_action_is_claimed() {
        let report = evaluate(&current_evidence_input());
        assert!(!report.terminal_head_preflight_ready_for_separate_device_lease);
        assert_eq!(
            report.status,
            "INCOMPLETE_QWEN80_TERMINAL_HEAD_UNREGISTERED_METAL_PREFLIGHT_REQUIRES_REAL_POST48_HIDDEN_AND_DURABLE_LEASE_CAPTURE"
        );
        assert!(report.staged_shader_and_plan_valid);
        assert!(!report.metal_context_or_dispatch_performed);
        assert!(!report.shader_registered_in_metal_mod);
        assert!(!report.complete_decoder_readiness_earned);
    }

    #[test]
    fn rejects_synthetic_or_wrong_shaped_post48_hidden_buffer() {
        let mut input = full_future_input();
        let hidden = input.post48_hidden_buffer.as_mut().unwrap();
        hidden.shape = vec![HIDDEN - 1];
        hidden.synthetic_or_component_fixture = true;
        let report = evaluate(&input);
        assert!(!report.terminal_head_preflight_ready_for_separate_device_lease);
        assert!(report
            .contract_errors
            .iter()
            .any(|error| error.contains("post-48-layer [2048]")));
    }

    #[test]
    fn rejects_capture_order_or_tail_before_all_rows() {
        let mut input = full_future_input();
        input
            .terminal_capture
            .as_mut()
            .unwrap()
            .ordered_stages
            .swap(2, 3);
        let report = evaluate(&input);
        assert!(!report.terminal_head_preflight_ready_for_separate_device_lease);
        assert!(report
            .contract_errors
            .iter()
            .any(|error| error.contains("terminal capture order")));
    }

    #[test]
    fn rejects_unsealed_baseline_or_non_durable_lease() {
        let mut input = full_future_input();
        input.sealed_cpu_baseline.as_mut().unwrap().seal_sha256 = "0".repeat(64);
        input
            .capture_lease
            .as_mut()
            .unwrap()
            .terminal_receipt_written_last = false;
        let report = evaluate(&input);
        assert!(!report.terminal_head_preflight_ready_for_separate_device_lease);
        assert!(report
            .contract_errors
            .iter()
            .any(|error| error.contains("sealed and integrity-verified")));
        assert!(report
            .contract_errors
            .iter()
            .any(|error| error.contains("receipt-last")));
    }

    #[test]
    fn rejects_selected_row_fallback_or_reserved_feedback() {
        let mut input = full_future_input();
        let capture = input.terminal_capture.as_mut().unwrap();
        capture.selected_row_shortcut_used = true;
        capture.fallback_used = true;
        capture.feedback_token_id = FIRST_RESERVED_ID;
        let report = evaluate(&input);
        assert!(!report.terminal_head_preflight_ready_for_separate_device_lease);
        assert!(report
            .contract_errors
            .iter()
            .any(|error| error.contains("selected-row shortcut")));
        assert!(report
            .contract_errors
            .iter()
            .any(|error| error.contains("deterministically sample")));
    }

    #[test]
    fn exact_hypothetical_capture_only_earns_narrow_terminal_frontier() {
        let report = evaluate(&full_future_input());
        assert!(report.terminal_head_preflight_ready_for_separate_device_lease);
        assert!(!report.complete_decoder_readiness_earned);
        assert!(!report.real_gravity_server_launch_precondition_satisfied);
        assert!(!report.metal_context_or_dispatch_performed);
    }
}

//! CPU-only Qwen3-Coder-Next terminal-head future-device contract.
//!
//! This target consumes the two already-written Qwen80 component receipts:
//! the direct-packed terminal-head CPU receipt and the source-bound
//! tokenizer/sampler receipt. It deliberately does not open a tensor payload,
//! inspect a live runtime, create a Metal context, start a server, invoke
//! HCLI, or measure performance.
//!
//! The output is a fail-closed host-dispatch/ABI ledger for a future device
//! implementation only:
//!
//! all-layer hidden [2048] -> final RMSNorm -> all 151936 lm-head rows ->
//! mask 151669..151935 -> deterministic sample -> validated feedback.
//!
//! The current component receipts are intentionally raw, unsealed CPU
//! evidence. They are useful provenance, but they cannot authorize a device
//! dispatch. A future executor must first bind them in a separately sealed
//! CPU-baseline envelope and then supply an actual source-bound all-layer
//! hidden vector with device parity. Until then this is NOT_RUNTIME,
//! NO_TOKEN, NO_HCLI, and NO_TPS.

use serde::Serialize;
use serde_json::{Map, Value};
use sha2::{Digest, Sha256};
use std::env;
use std::error::Error;
use std::fs;
use std::path::{Path, PathBuf};

const SCHEMA: &str = "hawking.ascension.qwen80_terminal_head_device_contract.v1";
const FUTURE_LEDGER_SCHEMA: &str =
    "hawking.ascension.qwen80_terminal_head_future_device_dispatch_ledger.v1";
const DECODER_READINESS_RESULT_SCHEMA: &str =
    "hawking.ascension.qwen80_complete_decoder_readiness_result.v1";
const SEALED_BASELINE_SCHEMA: &str =
    "hawking.ascension.qwen80_terminal_head_cpu_baseline_wrapper.v1";
const SEALED_BASELINE_STATUS: &str =
    "SEALED_CURRENT_ADMITTED_QWEN80_TERMINAL_HEAD_AND_SAMPLER_CPU_BASELINE";
const MODEL_ID: &str = "Qwen3-Coder-Next-80B";
const MODEL_KEY: &str = "qwen80";
const SOURCE_REPOSITORY: &str = "Qwen/Qwen3-Coder-Next";
const SOURCE_REVISION: &str = "a7fbcb5c0e12d62a448eaa0e260346bf5dcc0feb";
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
const FINAL_NORM_NAME: &str = "model.norm.weight";
const LM_HEAD_NAME: &str = "lm_head.weight";
const FINAL_NORM_ARTIFACT_SHA256: &str =
    "6306499804d27e48f0a041e94d366feae5cbf8436fac15815a559a15717ef36e";
const LM_HEAD_ARTIFACT_SHA256: &str =
    "549c448be683ed00ec792329c5167f3f0cacfcb3af339a1fb064ed0a004d9998";
const PACKED_MAGIC: &str = "HQ30G1B1";
const PACKED_VERSION: usize = 1;
const HIDDEN: usize = 2_048;
const LM_HEAD_VOCAB: usize = 151_936;
const TOKENIZER_VOCAB: usize = 151_669;
const RESERVED_TAIL_ROWS: usize = LM_HEAD_VOCAB - TOKENIZER_VOCAB;
const FIRST_RESERVED_ID: usize = TOKENIZER_VOCAB;
const LAST_RESERVED_ID: usize = LM_HEAD_VOCAB - 1;
const GROUP_SIZE: usize = 128;
const RMS_EPSILON_BITS: u64 = 897_988_541;
const DETERMINISTIC_SAMPLER: &str = "greedy_argmax_lowest_token_id_tie_break";

#[derive(Clone, Debug, Eq, PartialEq, Serialize)]
struct SourceBinding {
    model_id: String,
    model_key: String,
    source_repository: String,
    source_revision: String,
    manifest_seal_sha256: String,
    admission_receipt_seal_sha256: String,
    source_config_sha256: String,
    source_tokenizer_sha256: String,
    source_tokenizer_config_sha256: String,
    source_chat_template_sha256: String,
    source_generation_config_sha256: String,
}

impl SourceBinding {
    fn exact() -> Self {
        Self {
            model_id: MODEL_ID.into(),
            model_key: MODEL_KEY.into(),
            source_repository: SOURCE_REPOSITORY.into(),
            source_revision: SOURCE_REVISION.into(),
            manifest_seal_sha256: MANIFEST_SEAL.into(),
            admission_receipt_seal_sha256: ADMISSION_RECEIPT_SEAL.into(),
            source_config_sha256: SOURCE_CONFIG_SHA256.into(),
            source_tokenizer_sha256: SOURCE_TOKENIZER_SHA256.into(),
            source_tokenizer_config_sha256: SOURCE_TOKENIZER_CONFIG_SHA256.into(),
            source_chat_template_sha256: SOURCE_CHAT_TEMPLATE_SHA256.into(),
            source_generation_config_sha256: SOURCE_GENERATION_CONFIG_SHA256.into(),
        }
    }

    fn validate_exact(&self, label: &str) -> Result<(), String> {
        if self != &Self::exact() {
            return Err(format!("{label} source/config/tokenizer binding drifted"));
        }
        Ok(())
    }
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize)]
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
            shape: vec![LM_HEAD_VOCAB, HIDDEN],
            group_size: GROUP_SIZE,
            magic: PACKED_MAGIC.into(),
            version: PACKED_VERSION,
            artifact_sha256: LM_HEAD_ARTIFACT_SHA256.into(),
        }
    }

    fn validate_final_norm(&self) -> Result<(), String> {
        if self != &Self::final_norm() {
            return Err(
                "final RMSNorm ABI must remain direct-packed model.norm.weight [2048], group128"
                    .into(),
            );
        }
        Ok(())
    }

    fn validate_lm_head(&self) -> Result<(), String> {
        if self != &Self::lm_head() {
            return Err(
                "lm_head ABI must remain direct-packed lm_head.weight [151936,2048], group128"
                    .into(),
            );
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

#[derive(Clone, Debug)]
struct LoadedReceipt {
    evidence: FileEvidence,
    document: Value,
}

#[derive(Clone, Debug)]
struct CurrentCpuComponents {
    source: SourceBinding,
    final_norm: PackedTensorAbi,
    lm_head: PackedTensorAbi,
    terminal_receipt: FileEvidence,
    terminal_unsealed_preimage_sha256: String,
    tokenizer_receipt: FileEvidence,
    tokenizer_unsealed_preimage_sha256: String,
}

#[derive(Clone, Debug)]
struct SealedCpuBaseline {
    schema: String,
    status: String,
    seal_sha256: Option<String>,
    integrity_verified: bool,
    source: SourceBinding,
    terminal_receipt_document_sha256: String,
    terminal_receipt_unsealed_preimage_sha256: String,
    tokenizer_receipt_document_sha256: String,
    tokenizer_receipt_unsealed_preimage_sha256: String,
}

impl SealedCpuBaseline {
    fn validate(&self, current: &CurrentCpuComponents) -> Result<(), String> {
        if self.schema != SEALED_BASELINE_SCHEMA || self.status != SEALED_BASELINE_STATUS {
            return Err("future terminal-head CPU baseline wrapper schema/status drifted".into());
        }
        let seal = self
            .seal_sha256
            .as_deref()
            .ok_or("future terminal-head device ledger refuses an unsealed CPU baseline")?;
        if !is_lower_sha256(seal) || !self.integrity_verified {
            return Err(
                "future terminal-head device ledger requires a cryptographically verified CPU-baseline seal"
                    .into(),
            );
        }
        self.source.validate_exact("future sealed baseline")?;
        if self.source != current.source
            || self.terminal_receipt_document_sha256 != current.terminal_receipt.document_sha256
            || self.terminal_receipt_unsealed_preimage_sha256
                != current.terminal_unsealed_preimage_sha256
            || self.tokenizer_receipt_document_sha256 != current.tokenizer_receipt.document_sha256
            || self.tokenizer_receipt_unsealed_preimage_sha256
                != current.tokenizer_unsealed_preimage_sha256
        {
            return Err(
                "future terminal-head CPU baseline does not bind the exact current component evidence"
                    .into(),
            );
        }
        Ok(())
    }
}

#[derive(Clone, Debug)]
struct FullLayerHiddenInput {
    source: SourceBinding,
    shape: Vec<usize>,
    produced_by_all_48_layers: bool,
    synthetic_or_component_fixture: bool,
    device_parity_receipt_seal_sha256: Option<String>,
    fallback_used: bool,
}

impl FullLayerHiddenInput {
    fn validate(&self) -> Result<(), String> {
        self.source.validate_exact("all-layer hidden input")?;
        if self.shape.as_slice() != [HIDDEN] {
            return Err("all-layer hidden input must have exact shape [2048]".into());
        }
        if !self.produced_by_all_48_layers || self.synthetic_or_component_fixture {
            return Err(
                "future terminal-head device ledger requires an actual non-synthetic all-48-layer hidden input"
                    .into(),
            );
        }
        let parity = self
            .device_parity_receipt_seal_sha256
            .as_deref()
            .ok_or("future terminal-head device ledger requires a sealed device-parity receipt")?;
        if !is_lower_sha256(parity) || self.fallback_used {
            return Err(
                "future terminal-head device ledger rejects missing parity or a CPU/BF16 fallback"
                    .into(),
            );
        }
        Ok(())
    }
}

#[derive(Clone, Debug)]
struct DeviceExecutionPolicy {
    backend: String,
    strict_math: bool,
    cpu_fallback_used: bool,
    bf16_or_decoded_weight_fallback_used: bool,
    selected_row_head_shortcut_used: bool,
    timing_or_tps_allowed: bool,
}

impl DeviceExecutionPolicy {
    fn strict_future_component() -> Self {
        Self {
            backend: "metal".into(),
            strict_math: true,
            cpu_fallback_used: false,
            bf16_or_decoded_weight_fallback_used: false,
            selected_row_head_shortcut_used: false,
            timing_or_tps_allowed: false,
        }
    }

    fn validate(&self) -> Result<(), String> {
        if self.backend != "metal"
            || !self.strict_math
            || self.cpu_fallback_used
            || self.bf16_or_decoded_weight_fallback_used
            || self.selected_row_head_shortcut_used
            || self.timing_or_tps_allowed
        {
            return Err(
                "future terminal-head device ledger rejects fallback, selected-row, non-strict, or timing/TPS execution"
                    .into(),
            );
        }
        Ok(())
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum DispatchStage {
    Created,
    AllLayerHiddenBound,
    FinalRmsNormEncoded,
    AllRowHeadEncoded,
    ReservedTailMasked,
    DeterministicallySampled,
    FeedbackValidated,
}

#[derive(Clone, Debug)]
struct FutureDeviceHostDispatch {
    source: SourceBinding,
    final_norm: PackedTensorAbi,
    lm_head: PackedTensorAbi,
    policy: DeviceExecutionPolicy,
    stage: DispatchStage,
    operations: Vec<&'static str>,
}

impl FutureDeviceHostDispatch {
    fn begin(
        baseline: &SealedCpuBaseline,
        current: &CurrentCpuComponents,
        policy: DeviceExecutionPolicy,
    ) -> Result<Self, String> {
        baseline.validate(current)?;
        policy.validate()?;
        current.source.validate_exact("current CPU components")?;
        current.final_norm.validate_final_norm()?;
        current.lm_head.validate_lm_head()?;
        Ok(Self {
            source: current.source.clone(),
            final_norm: current.final_norm.clone(),
            lm_head: current.lm_head.clone(),
            policy,
            stage: DispatchStage::Created,
            operations: Vec::new(),
        })
    }

    fn assert_policy(&self) -> Result<(), String> {
        self.policy.validate()
    }

    fn bind_actual_all_layer_hidden(
        &mut self,
        hidden: &FullLayerHiddenInput,
    ) -> Result<(), String> {
        self.assert_policy()?;
        if self.stage != DispatchStage::Created {
            return Err(
                "all-layer hidden input must be bound before terminal-head dispatch".into(),
            );
        }
        hidden.validate()?;
        if hidden.source != self.source {
            return Err(
                "all-layer hidden source binding differs from terminal-head baseline".into(),
            );
        }
        self.stage = DispatchStage::AllLayerHiddenBound;
        self.operations.push("bind_actual_all_layer_hidden_[2048]");
        Ok(())
    }

    fn encode_final_rms_norm(&mut self, norm: &PackedTensorAbi) -> Result<(), String> {
        self.assert_policy()?;
        if self.stage != DispatchStage::AllLayerHiddenBound {
            return Err(
                "final RMSNorm must execute immediately after actual all-layer hidden input binding"
                    .into(),
            );
        }
        norm.validate_final_norm()?;
        if norm != &self.final_norm {
            return Err("future final RMSNorm binding differs from sealed CPU baseline".into());
        }
        self.stage = DispatchStage::FinalRmsNormEncoded;
        self.operations
            .push("final_rms_norm_model.norm.weight_[2048]_epsilon_1e-6");
        Ok(())
    }

    fn encode_all_row_lm_head(
        &mut self,
        head: &PackedTensorAbi,
        rows: usize,
    ) -> Result<(), String> {
        self.assert_policy()?;
        if self.stage != DispatchStage::FinalRmsNormEncoded {
            return Err(
                "all-row lm_head must execute after final RMSNorm and before tail masking".into(),
            );
        }
        head.validate_lm_head()?;
        if head != &self.lm_head || rows != LM_HEAD_VOCAB {
            return Err(
                "future terminal head requires exact baseline lm_head and all 151936 rows; selected rows are forbidden"
                    .into(),
            );
        }
        self.stage = DispatchStage::AllRowHeadEncoded;
        self.operations
            .push("all_row_lm_head_[151936,2048]_direct_packed_group128");
        Ok(())
    }

    fn mask_reserved_tail(
        &mut self,
        first_reserved_id: usize,
        reserved_rows: usize,
    ) -> Result<(), String> {
        self.assert_policy()?;
        if self.stage != DispatchStage::AllRowHeadEncoded {
            return Err(
                "reserved tail mask must execute after the all-row lm_head and before any sampler"
                    .into(),
            );
        }
        if first_reserved_id != FIRST_RESERVED_ID || reserved_rows != RESERVED_TAIL_ROWS {
            return Err("reserved tail mask must be exactly IDs 151669..151935 (267 rows)".into());
        }
        self.stage = DispatchStage::ReservedTailMasked;
        self.operations
            .push("mask_logits_ids_151669_through_151935_to_negative_infinity");
        Ok(())
    }

    fn deterministic_sample(
        &mut self,
        policy: &str,
        sampled_token_id: usize,
    ) -> Result<(), String> {
        self.assert_policy()?;
        if self.stage != DispatchStage::ReservedTailMasked {
            return Err(
                "sampling or tail feedback before the reserved-tail mask is forbidden".into(),
            );
        }
        if policy != DETERMINISTIC_SAMPLER {
            return Err(
                "future terminal-head sampler policy drifted from deterministic contract".into(),
            );
        }
        if sampled_token_id >= TOKENIZER_VOCAB {
            return Err(
                "future terminal-head sampler produced a reserved/unaddressable token ID".into(),
            );
        }
        self.stage = DispatchStage::DeterministicallySampled;
        self.operations
            .push("deterministic_sample_after_complete_tail_mask");
        Ok(())
    }

    fn validate_feedback(&mut self, feedback_token_id: usize) -> Result<(), String> {
        self.assert_policy()?;
        if self.stage != DispatchStage::DeterministicallySampled {
            return Err("autoregressive feedback must follow deterministic sampling".into());
        }
        if feedback_token_id >= TOKENIZER_VOCAB {
            return Err(
                "reserved lm_head tail token cannot enter tokenizer/embedding feedback".into(),
            );
        }
        self.stage = DispatchStage::FeedbackValidated;
        self.operations
            .push("validate_tokenizer_addressable_feedback_before_next_state_step");
        Ok(())
    }

    fn finish(self) -> Result<Vec<&'static str>, String> {
        self.policy.validate()?;
        if self.stage != DispatchStage::FeedbackValidated {
            return Err("terminal-head host-dispatch ledger is incomplete".into());
        }
        Ok(self.operations)
    }
}

#[derive(Serialize)]
struct CurrentCpuComponentsReport {
    terminal_head_cpu_receipt: FileEvidence,
    terminal_head_cpu_receipt_unsealed_preimage_sha256: String,
    tokenizer_sampler_receipt: FileEvidence,
    tokenizer_sampler_receipt_unsealed_preimage_sha256: String,
    source_binding: SourceBinding,
    final_norm_abi: PackedTensorAbi,
    lm_head_abi: PackedTensorAbi,
    tokenizer_addressable_vocab_size: usize,
    lm_head_vocab_size: usize,
    reserved_tail_first_id: usize,
    reserved_tail_last_id: usize,
    reserved_tail_rows: usize,
    raw_component_receipts_are_unsealed: bool,
}

#[derive(Serialize)]
struct FutureDeviceLedgerReport {
    schema: &'static str,
    state: &'static str,
    dispatch_authorized_now: bool,
    required_sealed_cpu_baseline_wrapper_schema: &'static str,
    raw_component_receipts_can_authorize_device_dispatch: bool,
    actual_all_layer_hidden_input_available: bool,
    actual_device_parity_available: bool,
    backend_required: &'static str,
    strict_math_required: bool,
    cpu_or_bf16_fallback_allowed: bool,
    selected_row_lm_head_allowed: bool,
    timing_tps_or_tg_allowed: bool,
    ordered_host_dispatch: Vec<&'static str>,
    exact_tail_mask: Vec<usize>,
}

#[derive(Serialize)]
struct DecoderReadinessIntegrationReport {
    target_readiness_result_schema: &'static str,
    terminal_head_operator_classes: Vec<&'static str>,
    source_component_receipt_mapping: Vec<&'static str>,
    current_cpu_components_are_full_path_device_ledgers: bool,
    eligible_to_emit_promotable_readiness_descriptors_now: bool,
    required_before_promotable_descriptor: Vec<&'static str>,
    claim_boundary: &'static str,
}

#[derive(Serialize)]
struct RejectionReport {
    wrong_final_norm_shape_rejected: bool,
    wrong_lm_head_shape_rejected: bool,
    wrong_direct_group_size_rejected: bool,
    all_row_head_before_rms_norm_rejected: bool,
    tail_mask_before_all_row_head_rejected: bool,
    sample_or_feedback_before_tail_mask_rejected: bool,
    unsealed_cpu_baseline_rejected: bool,
    source_mismatch_rejected: bool,
    fallback_rejected: bool,
    missing_actual_all_layer_hidden_or_device_parity_rejected: bool,
}

#[derive(Serialize)]
struct Report {
    schema: &'static str,
    status: &'static str,
    component_contract_only: bool,
    terminal_head_cpu_components: CurrentCpuComponentsReport,
    future_device_dispatch_ledger: FutureDeviceLedgerReport,
    decoder_readiness_integration: DecoderReadinessIntegrationReport,
    rejection_tests: RejectionReport,
    rawls_hybrid_scheduler_handoff: Vec<&'static str>,
    claim_boundary: Vec<&'static str>,
    live_artifact_scan_performed: bool,
    metal_device_or_dispatch_performed: bool,
    model_runtime_or_server_started: bool,
    hcli_execution_performed: bool,
    tps_or_tg_measurement_performed: bool,
    unsealed_preimage_sha256: String,
}

struct Args {
    terminal_head_cpu_receipt: PathBuf,
    tokenizer_sampler_receipt: PathBuf,
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

fn object<'a>(
    value: &'a Value,
    field: &str,
    label: &str,
) -> Result<&'a Map<String, Value>, String> {
    value
        .get(field)
        .and_then(Value::as_object)
        .ok_or_else(|| format!("{label} missing object {field:?}"))
}

fn required_string<'a>(
    value: &'a Map<String, Value>,
    field: &str,
    label: &str,
) -> Result<&'a str, String> {
    value
        .get(field)
        .and_then(Value::as_str)
        .filter(|item| !item.is_empty())
        .ok_or_else(|| format!("{label} missing non-empty string {field:?}"))
}

fn required_usize(value: &Map<String, Value>, field: &str, label: &str) -> Result<usize, String> {
    value
        .get(field)
        .and_then(Value::as_u64)
        .and_then(|item| usize::try_from(item).ok())
        .ok_or_else(|| format!("{label} missing usize {field:?}"))
}

fn required_bool(
    value: &Map<String, Value>,
    field: &str,
    expected: bool,
    label: &str,
) -> Result<(), String> {
    if value.get(field).and_then(Value::as_bool) != Some(expected) {
        return Err(format!("{label} field {field:?} must be {expected}"));
    }
    Ok(())
}

fn required_string_eq(
    value: &Map<String, Value>,
    field: &str,
    expected: &str,
    label: &str,
) -> Result<(), String> {
    if required_string(value, field, label)? != expected {
        return Err(format!("{label} field {field:?} drifted"));
    }
    Ok(())
}

fn required_usize_eq(
    value: &Map<String, Value>,
    field: &str,
    expected: usize,
    label: &str,
) -> Result<(), String> {
    if required_usize(value, field, label)? != expected {
        return Err(format!("{label} field {field:?} drifted"));
    }
    Ok(())
}

fn array_usizes(
    value: &Map<String, Value>,
    field: &str,
    label: &str,
) -> Result<Vec<usize>, String> {
    value
        .get(field)
        .and_then(Value::as_array)
        .ok_or_else(|| format!("{label} missing array {field:?}"))?
        .iter()
        .map(|item| {
            item.as_u64()
                .and_then(|number| usize::try_from(number).ok())
                .ok_or_else(|| format!("{label} has a non-usize in {field:?}"))
        })
        .collect()
}

fn regular_json(path: &Path, label: &str) -> Result<LoadedReceipt, String> {
    if !path.is_absolute() {
        return Err(format!("{label} must be an absolute path"));
    }
    let metadata = fs::symlink_metadata(path)
        .map_err(|error| format!("{label} metadata failed at {}: {error}", path.display()))?;
    if !metadata.file_type().is_file() || metadata.file_type().is_symlink() {
        return Err(format!("{label} must be a regular non-symlink file"));
    }
    let path = fs::canonicalize(path)
        .map_err(|error| format!("{label} canonicalize failed at {}: {error}", path.display()))?;
    let bytes = fs::read(&path)
        .map_err(|error| format!("{label} read failed at {}: {error}", path.display()))?;
    let document: Value =
        serde_json::from_slice(&bytes).map_err(|error| format!("{label} invalid JSON: {error}"))?;
    if !document.is_object() {
        return Err(format!("{label} root must be a JSON object"));
    }
    Ok(LoadedReceipt {
        evidence: FileEvidence {
            path: path.display().to_string(),
            bytes: bytes.len(),
            document_sha256: sha256_hex(&bytes),
        },
        document,
    })
}

fn parsed_source_from_terminal(binding: &Map<String, Value>) -> Result<SourceBinding, String> {
    required_string_eq(
        binding,
        "manifest_seal_sha256",
        MANIFEST_SEAL,
        "terminal artifact binding",
    )?;
    required_string_eq(
        binding,
        "admission_receipt_seal_sha256",
        ADMISSION_RECEIPT_SEAL,
        "terminal artifact binding",
    )?;
    required_string_eq(
        binding,
        "source_repository",
        SOURCE_REPOSITORY,
        "terminal artifact binding",
    )?;
    required_string_eq(
        binding,
        "source_revision",
        SOURCE_REVISION,
        "terminal artifact binding",
    )?;
    required_string_eq(
        binding,
        "source_config_sha256",
        SOURCE_CONFIG_SHA256,
        "terminal artifact binding",
    )?;
    required_string_eq(
        binding,
        "source_tokenizer_sha256",
        SOURCE_TOKENIZER_SHA256,
        "terminal artifact binding",
    )?;
    Ok(SourceBinding::exact())
}

fn parse_tensor_abi(
    artifact: &Map<String, Value>,
    expected: &PackedTensorAbi,
    label: &str,
) -> Result<PackedTensorAbi, String> {
    let header = artifact
        .get("header")
        .and_then(Value::as_object)
        .ok_or_else(|| format!("{label} missing direct-packed header"))?;
    let observed = PackedTensorAbi {
        name: required_string(artifact, "name", label)?.to_owned(),
        shape: array_usizes(header, "shape", label)?,
        group_size: required_usize(header, "group_size", label)?,
        magic: required_string(header, "magic", label)?.to_owned(),
        version: required_usize(header, "version", label)?,
        artifact_sha256: required_string(artifact, "artifact_sha256", label)?.to_owned(),
    };
    if &observed != expected {
        return Err(format!("{label} direct-packed tensor ABI drifted"));
    }
    Ok(observed)
}

fn require_true_fields(
    object: &Map<String, Value>,
    fields: &[&str],
    label: &str,
) -> Result<(), String> {
    for field in fields {
        required_bool(object, field, true, label)?;
    }
    Ok(())
}

fn validate_terminal_receipt(
    receipt: &LoadedReceipt,
) -> Result<(SourceBinding, PackedTensorAbi, PackedTensorAbi, String), String> {
    if receipt.evidence.document_sha256 != TERMINAL_RECEIPT_DOCUMENT_SHA256 {
        return Err(
            "terminal-head CPU receipt bytes are not the pinned current component receipt".into(),
        );
    }
    let document = receipt
        .document
        .as_object()
        .ok_or("terminal-head CPU receipt root is not an object")?;
    required_string_eq(
        document,
        "schema",
        TERMINAL_RECEIPT_SCHEMA,
        "terminal-head CPU receipt",
    )?;
    required_string_eq(
        document,
        "status",
        TERMINAL_RECEIPT_STATUS,
        "terminal-head CPU receipt",
    )?;
    if document.contains_key("seal_sha256") {
        return Err(
            "terminal-head CPU component receipt is expected to be raw/unsealed evidence".into(),
        );
    }
    required_bool(
        document,
        "component_only",
        true,
        "terminal-head CPU receipt",
    )?;
    required_bool(
        document,
        "complete_artifact_scan_performed",
        false,
        "terminal-head CPU receipt",
    )?;
    required_bool(
        document,
        "metal_device_or_dispatch_performed",
        false,
        "terminal-head CPU receipt",
    )?;
    let binding = object(
        &receipt.document,
        "artifact_binding",
        "terminal-head CPU receipt",
    )?;
    let source = parsed_source_from_terminal(binding)?;
    required_usize_eq(binding, "hidden_size", HIDDEN, "terminal artifact binding")?;
    required_usize_eq(
        binding,
        "lm_head_vocab_size",
        LM_HEAD_VOCAB,
        "terminal artifact binding",
    )?;
    required_usize_eq(
        binding,
        "tokenizer_vocab_size",
        TOKENIZER_VOCAB,
        "terminal artifact binding",
    )?;
    required_usize_eq(
        binding,
        "reserved_lm_head_tail_rows",
        RESERVED_TAIL_ROWS,
        "terminal artifact binding",
    )?;
    if binding.get("rms_epsilon_bits").and_then(Value::as_u64) != Some(RMS_EPSILON_BITS) {
        return Err("terminal artifact binding RMS epsilon drifted".into());
    }
    let final_norm = parse_tensor_abi(
        object(
            &receipt.document,
            "artifact_binding",
            "terminal-head CPU receipt",
        )?
        .get("final_norm")
        .and_then(Value::as_object)
        .ok_or("terminal artifact binding missing final_norm")?,
        &PackedTensorAbi::final_norm(),
        "terminal final_norm",
    )?;
    let lm_head = parse_tensor_abi(
        object(
            &receipt.document,
            "artifact_binding",
            "terminal-head CPU receipt",
        )?
        .get("lm_head")
        .and_then(Value::as_object)
        .ok_or("terminal artifact binding missing lm_head")?,
        &PackedTensorAbi::lm_head(),
        "terminal lm_head",
    )?;
    let tail = object(
        &receipt.document,
        "reserved_tail_mask_and_sampler",
        "terminal-head CPU receipt",
    )?;
    required_usize_eq(
        tail,
        "first_reserved_id",
        FIRST_RESERVED_ID,
        "terminal tail mask",
    )?;
    required_usize_eq(
        tail,
        "last_reserved_id",
        LAST_RESERVED_ID,
        "terminal tail mask",
    )?;
    required_usize_eq(
        tail,
        "reserved_rows_masked",
        RESERVED_TAIL_ROWS,
        "terminal tail mask",
    )?;
    require_true_fields(
        tail,
        &[
            "every_reserved_logit_negative_infinity",
            "sampled_token_is_tokenizer_addressable",
        ],
        "terminal tail mask",
    )?;
    let rejections = object(
        &receipt.document,
        "rejection_tests",
        "terminal-head CPU receipt",
    )?;
    require_true_fields(
        rejections,
        &[
            "wrong_lm_head_shape_rejected",
            "wrong_direct_group_size_rejected",
            "wrong_tail_partition_rejected",
            "tail_mask_wrong_cutoff_rejected",
            "unmasked_sampler_api_unavailable",
        ],
        "terminal-head CPU receipt rejections",
    )?;
    let preimage = required_string(
        document,
        "unsealed_preimage_sha256",
        "terminal-head CPU receipt",
    )?;
    if preimage != TERMINAL_RECEIPT_UNSEALED_PREIMAGE_SHA256 || !is_lower_sha256(preimage) {
        return Err("terminal-head CPU receipt unsealed preimage drifted".into());
    }
    Ok((source, final_norm, lm_head, preimage.to_owned()))
}

fn validate_tokenizer_receipt(receipt: &LoadedReceipt) -> Result<(SourceBinding, String), String> {
    if receipt.evidence.document_sha256 != TOKENIZER_RECEIPT_DOCUMENT_SHA256 {
        return Err(
            "tokenizer/sampler receipt bytes are not the pinned current component receipt".into(),
        );
    }
    let document = receipt
        .document
        .as_object()
        .ok_or("tokenizer/sampler receipt root is not an object")?;
    required_string_eq(
        document,
        "schema",
        TOKENIZER_RECEIPT_SCHEMA,
        "tokenizer/sampler receipt",
    )?;
    required_string_eq(
        document,
        "status",
        TOKENIZER_RECEIPT_STATUS,
        "tokenizer/sampler receipt",
    )?;
    if document.contains_key("seal_sha256") {
        return Err(
            "tokenizer/sampler component receipt is expected to be raw/unsealed evidence".into(),
        );
    }
    required_bool(
        document,
        "component_only",
        true,
        "tokenizer/sampler receipt",
    )?;
    required_bool(
        document,
        "live_packed_artifact_scan_performed",
        false,
        "tokenizer/sampler receipt",
    )?;
    required_bool(
        document,
        "metal_device_or_dispatch_performed",
        false,
        "tokenizer/sampler receipt",
    )?;
    let binding = object(
        &receipt.document,
        "source_binding",
        "tokenizer/sampler receipt",
    )?;
    required_string_eq(
        binding,
        "source_repository",
        SOURCE_REPOSITORY,
        "tokenizer source binding",
    )?;
    required_string_eq(
        binding,
        "source_revision_from_pre_admitted_source_audit",
        SOURCE_REVISION,
        "tokenizer source binding",
    )?;
    required_string_eq(
        binding,
        "config_sha256",
        SOURCE_CONFIG_SHA256,
        "tokenizer source binding",
    )?;
    required_string_eq(
        binding,
        "tokenizer_sha256",
        SOURCE_TOKENIZER_SHA256,
        "tokenizer source binding",
    )?;
    required_string_eq(
        binding,
        "tokenizer_config_sha256",
        SOURCE_TOKENIZER_CONFIG_SHA256,
        "tokenizer source binding",
    )?;
    required_string_eq(
        binding,
        "chat_template_sha256",
        SOURCE_CHAT_TEMPLATE_SHA256,
        "tokenizer source binding",
    )?;
    required_string_eq(
        binding,
        "generation_config_sha256",
        SOURCE_GENERATION_CONFIG_SHA256,
        "tokenizer source binding",
    )?;
    required_string_eq(
        binding,
        "architecture",
        "Qwen3NextForCausalLM",
        "tokenizer source binding",
    )?;
    required_string_eq(
        binding,
        "model_type",
        "qwen3_next",
        "tokenizer source binding",
    )?;
    required_usize_eq(
        binding,
        "lm_head_vocab_size",
        LM_HEAD_VOCAB,
        "tokenizer source binding",
    )?;
    required_usize_eq(
        binding,
        "tokenizer_addressable_vocab_size",
        TOKENIZER_VOCAB,
        "tokenizer source binding",
    )?;
    required_usize_eq(
        binding,
        "reserved_lm_head_tail_rows",
        RESERVED_TAIL_ROWS,
        "tokenizer source binding",
    )?;
    require_true_fields(
        binding,
        &[
            "tokenizer_config_template_equivalent_after_exactly_one_terminal_lf_normalization",
            "bounded_one_user_branch_anchors_verified",
        ],
        "tokenizer source binding",
    )?;
    let sampler = object(
        &receipt.document,
        "sampler_fixture",
        "tokenizer/sampler receipt",
    )?;
    required_usize_eq(
        sampler,
        "reserved_tail_mask_cutoff",
        FIRST_RESERVED_ID,
        "tokenizer sampler fixture",
    )?;
    required_usize_eq(
        sampler,
        "sampled_feedback_token_id",
        TOKENIZER_VOCAB - 1,
        "tokenizer sampler fixture",
    )?;
    require_true_fields(
        sampler,
        &[
            "all_selected_reserved_fixture_logits_are_negative_infinity",
            "sampled_id_is_tokenizer_addressable",
        ],
        "tokenizer sampler fixture",
    )?;
    let rejections = object(
        &receipt.document,
        "rejection_tests",
        "tokenizer/sampler receipt",
    )?;
    require_true_fields(
        rejections,
        &[
            "source_config_hash_mismatch_rejected",
            "source_template_mismatch_rejected",
            "sample_before_tail_mask_rejected",
            "wrong_tail_mask_cutoff_rejected",
            "reserved_tail_feedback_rejected",
        ],
        "tokenizer/sampler receipt rejections",
    )?;
    let preimage = required_string(
        document,
        "unsealed_preimage_sha256",
        "tokenizer/sampler receipt",
    )?;
    if preimage != TOKENIZER_RECEIPT_UNSEALED_PREIMAGE_SHA256 || !is_lower_sha256(preimage) {
        return Err("tokenizer/sampler receipt unsealed preimage drifted".into());
    }
    Ok((SourceBinding::exact(), preimage.to_owned()))
}

fn load_current_cpu_components(
    terminal_head_cpu_receipt: &Path,
    tokenizer_sampler_receipt: &Path,
) -> Result<CurrentCpuComponents, String> {
    let terminal = regular_json(terminal_head_cpu_receipt, "terminal-head CPU receipt")?;
    let tokenizer = regular_json(tokenizer_sampler_receipt, "tokenizer/sampler receipt")?;
    let (terminal_source, final_norm, lm_head, terminal_preimage) =
        validate_terminal_receipt(&terminal)?;
    let (tokenizer_source, tokenizer_preimage) = validate_tokenizer_receipt(&tokenizer)?;
    if terminal_source != tokenizer_source {
        return Err(
            "terminal-head and tokenizer/sampler component receipts have different source bindings"
                .into(),
        );
    }
    Ok(CurrentCpuComponents {
        source: terminal_source,
        final_norm,
        lm_head,
        terminal_receipt: terminal.evidence,
        terminal_unsealed_preimage_sha256: terminal_preimage,
        tokenizer_receipt: tokenizer.evidence,
        tokenizer_unsealed_preimage_sha256: tokenizer_preimage,
    })
}

fn current_report(current: &CurrentCpuComponents) -> CurrentCpuComponentsReport {
    CurrentCpuComponentsReport {
        terminal_head_cpu_receipt: current.terminal_receipt.clone(),
        terminal_head_cpu_receipt_unsealed_preimage_sha256: current
            .terminal_unsealed_preimage_sha256
            .clone(),
        tokenizer_sampler_receipt: current.tokenizer_receipt.clone(),
        tokenizer_sampler_receipt_unsealed_preimage_sha256: current
            .tokenizer_unsealed_preimage_sha256
            .clone(),
        source_binding: current.source.clone(),
        final_norm_abi: current.final_norm.clone(),
        lm_head_abi: current.lm_head.clone(),
        tokenizer_addressable_vocab_size: TOKENIZER_VOCAB,
        lm_head_vocab_size: LM_HEAD_VOCAB,
        reserved_tail_first_id: FIRST_RESERVED_ID,
        reserved_tail_last_id: LAST_RESERVED_ID,
        reserved_tail_rows: RESERVED_TAIL_ROWS,
        raw_component_receipts_are_unsealed: true,
    }
}

fn future_device_ledger_report() -> FutureDeviceLedgerReport {
    FutureDeviceLedgerReport {
        schema: FUTURE_LEDGER_SCHEMA,
        state:
            "BLOCKED_UNTIL_SEALED_CPU_BASELINE_AND_ACTUAL_ALL_LAYER_HIDDEN_DEVICE_PARITY_NOT_RUNTIME_NO_TOKEN_NO_HCLI_NO_TPS",
        dispatch_authorized_now: false,
        required_sealed_cpu_baseline_wrapper_schema: SEALED_BASELINE_SCHEMA,
        raw_component_receipts_can_authorize_device_dispatch: false,
        actual_all_layer_hidden_input_available: false,
        actual_device_parity_available: false,
        backend_required: "metal",
        strict_math_required: true,
        cpu_or_bf16_fallback_allowed: false,
        selected_row_lm_head_allowed: false,
        timing_tps_or_tg_allowed: false,
        ordered_host_dispatch: vec![
            "verify a cryptographically sealed CPU-baseline wrapper binds both exact raw component receipt documents and preimages",
            "bind one actual source-bound, non-synthetic all-48-layer hidden vector [2048] with a sealed device-parity receipt",
            "encode final RMSNorm using model.norm.weight HQ30G1B1 [2048], group_size=128, epsilon=1e-6",
            "encode every lm_head.weight HQ30G1B1 row [151936,2048], group_size=128; no selected-row shortcut",
            "mask exactly logits 151669..151935 to negative infinity before sampler access",
            "deterministically sample with documented lowest-token-ID tie break and reject IDs >=151669",
            "validate the sampled feedback ID against tokenizer namespace before next embedding/state step",
        ],
        exact_tail_mask: vec![FIRST_RESERVED_ID, LAST_RESERVED_ID, RESERVED_TAIL_ROWS],
    }
}

fn decoder_readiness_integration_report() -> DecoderReadinessIntegrationReport {
    DecoderReadinessIntegrationReport {
        target_readiness_result_schema: DECODER_READINESS_RESULT_SCHEMA,
        terminal_head_operator_classes: vec![
            "final_norm",
            "lm_head",
            "tail_mask",
            "sampler",
            "tokenizer_template",
        ],
        source_component_receipt_mapping: vec![
            "direct-packed terminal-head CPU receipt supplies only CPU-component provenance for final_norm, lm_head, and tail_mask",
            "source tokenizer/sampler receipt supplies only CPU-component provenance for sampler and tokenizer_template",
        ],
        current_cpu_components_are_full_path_device_ledgers: false,
        eligible_to_emit_promotable_readiness_descriptors_now: false,
        required_before_promotable_descriptor: vec![
            "a cryptographically sealed CPU-baseline wrapper must bind both exact component receipt documents and preimages",
            "one actual source-bound non-synthetic all-48-layer hidden [2048] input must reach the terminal head",
            "strict-Metal final RMSNorm and all-151936-row lm_head parity must be sealed against that exact CPU baseline",
            "the same full-path device ledger must prove mask 151669..151935 before deterministic sampling and tokenizer-addressable feedback",
            "the resulting descriptor must declare source_bound=true, artifact_bound=true, full_path=true, complete_token_path=true, device_parity_passed=true, fixture_only=false, synthetic_input=false, component_only=false",
        ],
        claim_boundary:
            "This maps the terminal-head prerequisites into decoder-readiness operator classes only; it is not a readiness descriptor, decoder token, HCLI result, or TPS/TG evidence.",
    }
}

fn test_sha256(fill: char) -> String {
    std::iter::repeat_n(fill, 64).collect()
}

fn current_fixture() -> CurrentCpuComponents {
    CurrentCpuComponents {
        source: SourceBinding::exact(),
        final_norm: PackedTensorAbi::final_norm(),
        lm_head: PackedTensorAbi::lm_head(),
        terminal_receipt: FileEvidence {
            path: "/fixture/QWEN80_DIRECT_PACKED_TERMINAL_HEAD_CPU_COMPONENT_RECEIPT.json".into(),
            bytes: 1,
            document_sha256: TERMINAL_RECEIPT_DOCUMENT_SHA256.into(),
        },
        terminal_unsealed_preimage_sha256: TERMINAL_RECEIPT_UNSEALED_PREIMAGE_SHA256.into(),
        tokenizer_receipt: FileEvidence {
            path: "/fixture/QWEN80_TOKENIZER_TEMPLATE_SAMPLER_HANDOFF_COMPONENT_RECEIPT.json"
                .into(),
            bytes: 1,
            document_sha256: TOKENIZER_RECEIPT_DOCUMENT_SHA256.into(),
        },
        tokenizer_unsealed_preimage_sha256: TOKENIZER_RECEIPT_UNSEALED_PREIMAGE_SHA256.into(),
    }
}

fn sealed_baseline_fixture(current: &CurrentCpuComponents) -> SealedCpuBaseline {
    SealedCpuBaseline {
        schema: SEALED_BASELINE_SCHEMA.into(),
        status: SEALED_BASELINE_STATUS.into(),
        seal_sha256: Some(test_sha256('a')),
        integrity_verified: true,
        source: current.source.clone(),
        terminal_receipt_document_sha256: current.terminal_receipt.document_sha256.clone(),
        terminal_receipt_unsealed_preimage_sha256: current
            .terminal_unsealed_preimage_sha256
            .clone(),
        tokenizer_receipt_document_sha256: current.tokenizer_receipt.document_sha256.clone(),
        tokenizer_receipt_unsealed_preimage_sha256: current
            .tokenizer_unsealed_preimage_sha256
            .clone(),
    }
}

fn all_layer_hidden_fixture() -> FullLayerHiddenInput {
    FullLayerHiddenInput {
        source: SourceBinding::exact(),
        shape: vec![HIDDEN],
        produced_by_all_48_layers: true,
        synthetic_or_component_fixture: false,
        device_parity_receipt_seal_sha256: Some(test_sha256('b')),
        fallback_used: false,
    }
}

fn rejection_report() -> RejectionReport {
    let current = current_fixture();

    let mut wrong_norm = PackedTensorAbi::final_norm();
    wrong_norm.shape = vec![HIDDEN + 1];
    let wrong_final_norm_shape_rejected = wrong_norm.validate_final_norm().is_err();

    let mut wrong_head = PackedTensorAbi::lm_head();
    wrong_head.shape = vec![LM_HEAD_VOCAB - 1, HIDDEN];
    let wrong_lm_head_shape_rejected = wrong_head.validate_lm_head().is_err();

    let mut wrong_group = PackedTensorAbi::lm_head();
    wrong_group.group_size = 64;
    let wrong_direct_group_size_rejected = wrong_group.validate_lm_head().is_err();

    let baseline = sealed_baseline_fixture(&current);
    let policy = DeviceExecutionPolicy::strict_future_component();
    let mut wrong_order =
        FutureDeviceHostDispatch::begin(&baseline, &current, policy.clone()).unwrap();
    let all_row_head_before_rms_norm_rejected = wrong_order
        .encode_all_row_lm_head(&current.lm_head, LM_HEAD_VOCAB)
        .is_err();
    let tail_mask_before_all_row_head_rejected = wrong_order
        .mask_reserved_tail(FIRST_RESERVED_ID, RESERVED_TAIL_ROWS)
        .is_err();

    wrong_order
        .bind_actual_all_layer_hidden(&all_layer_hidden_fixture())
        .unwrap();
    wrong_order
        .encode_final_rms_norm(&current.final_norm)
        .unwrap();
    wrong_order
        .encode_all_row_lm_head(&current.lm_head, LM_HEAD_VOCAB)
        .unwrap();
    let sample_or_feedback_before_tail_mask_rejected = wrong_order
        .deterministic_sample(DETERMINISTIC_SAMPLER, 42)
        .is_err()
        && wrong_order.validate_feedback(42).is_err();

    let mut unsealed = sealed_baseline_fixture(&current);
    unsealed.seal_sha256 = None;
    let unsealed_cpu_baseline_rejected =
        FutureDeviceHostDispatch::begin(&unsealed, &current, policy.clone()).is_err();

    let mut mismatched_source = sealed_baseline_fixture(&current);
    mismatched_source.source.source_config_sha256 = test_sha256('c');
    let source_mismatch_rejected =
        FutureDeviceHostDispatch::begin(&mismatched_source, &current, policy.clone()).is_err();

    let mut fallback = DeviceExecutionPolicy::strict_future_component();
    fallback.cpu_fallback_used = true;
    let fallback_rejected = FutureDeviceHostDispatch::begin(&baseline, &current, fallback).is_err();

    let mut absent_hidden = all_layer_hidden_fixture();
    absent_hidden.device_parity_receipt_seal_sha256 = None;
    let mut hidden_dispatch = FutureDeviceHostDispatch::begin(&baseline, &current, policy).unwrap();
    let missing_actual_all_layer_hidden_or_device_parity_rejected = hidden_dispatch
        .bind_actual_all_layer_hidden(&absent_hidden)
        .is_err();

    RejectionReport {
        wrong_final_norm_shape_rejected,
        wrong_lm_head_shape_rejected,
        wrong_direct_group_size_rejected,
        all_row_head_before_rms_norm_rejected,
        tail_mask_before_all_row_head_rejected,
        sample_or_feedback_before_tail_mask_rejected,
        unsealed_cpu_baseline_rejected,
        source_mismatch_rejected,
        fallback_rejected,
        missing_actual_all_layer_hidden_or_device_parity_rejected,
    }
}

fn all_rejections_pass(report: &RejectionReport) -> bool {
    report.wrong_final_norm_shape_rejected
        && report.wrong_lm_head_shape_rejected
        && report.wrong_direct_group_size_rejected
        && report.all_row_head_before_rms_norm_rejected
        && report.tail_mask_before_all_row_head_rejected
        && report.sample_or_feedback_before_tail_mask_rejected
        && report.unsealed_cpu_baseline_rejected
        && report.source_mismatch_rejected
        && report.fallback_rejected
        && report.missing_actual_all_layer_hidden_or_device_parity_rejected
}

fn rawls_handoff() -> Vec<&'static str> {
    vec![
        "Do not register, compile, or dispatch this contract yet. First emit and cryptographically verify a new sealed terminal-head CPU-baseline wrapper that binds the exact raw terminal-head and tokenizer/sampler receipt document SHA-256 values plus their unsealed-preimage SHA-256 values.",
        "The future Metal host encoder must accept only an actual source-bound non-synthetic [2048] hidden vector after all 48 Qwen80 hybrid layers. It must bind the corresponding sealed device-parity receipt before final-head work.",
        "Use only direct-packed HQ30G1B1 model.norm.weight [2048] and lm_head.weight [151936,2048], both group_size=128. Preserve packed sign/scale interpretation; do not substitute decoded BF16, CPU, MPS, or selected-row weights.",
        "Encode final RMSNorm first with epsilon 1e-6, then all 151936 lm-head rows. The logit-domain tail mask must set every ID 151669..151935 to negative infinity before any sampler read.",
        "Implement deterministic argmax with lowest-token-ID tie breaking, reject sampled/feedback IDs >=151669, and validate tokenizer addressability before the next embedding/KV or DeltaNet-state step.",
        "Keep the eventual component capture strict-math, device-exclusive when parity-sensitive, non-timed, and explicitly component-only until the full decoder/token evidence is earned. It cannot report HCLI, BASE_TRUE_TPS, TG, capability, or tournament progress.",
    ]
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

fn usage() -> &'static str {
    "usage: ascension_qwen80_terminal_head_device_contract \
--terminal-head-cpu-receipt ABSOLUTE_PATH \
--tokenizer-sampler-receipt ABSOLUTE_PATH \
--out ABSOLUTE_PATH"
}

fn parse_args() -> Result<Args, Box<dyn Error>> {
    let mut terminal_head_cpu_receipt = None;
    let mut tokenizer_sampler_receipt = None;
    let mut out = None;
    let mut values = env::args().skip(1);
    while let Some(flag) = values.next() {
        let value = values
            .next()
            .ok_or_else(|| format!("missing value after {flag}; {}", usage()))?;
        let target = PathBuf::from(value);
        match flag.as_str() {
            "--terminal-head-cpu-receipt" => {
                if terminal_head_cpu_receipt.replace(target).is_some() {
                    return Err("--terminal-head-cpu-receipt repeated".into());
                }
            }
            "--tokenizer-sampler-receipt" => {
                if tokenizer_sampler_receipt.replace(target).is_some() {
                    return Err("--tokenizer-sampler-receipt repeated".into());
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
        terminal_head_cpu_receipt: terminal_head_cpu_receipt
            .ok_or_else(|| format!("missing --terminal-head-cpu-receipt; {}", usage()))?,
        tokenizer_sampler_receipt: tokenizer_sampler_receipt
            .ok_or_else(|| format!("missing --tokenizer-sampler-receipt; {}", usage()))?,
        out: out.ok_or_else(|| format!("missing --out; {}", usage()))?,
    };
    if !args.terminal_head_cpu_receipt.is_absolute()
        || !args.tokenizer_sampler_receipt.is_absolute()
        || !args.out.is_absolute()
    {
        return Err("all paths must be absolute".into());
    }
    Ok(args)
}

fn run(args: Args) -> Result<(), Box<dyn Error>> {
    let current = load_current_cpu_components(
        &args.terminal_head_cpu_receipt,
        &args.tokenizer_sampler_receipt,
    )?;
    let rejection_tests = rejection_report();
    if !all_rejections_pass(&rejection_tests) {
        return Err("terminal-head future-device contract rejection tests did not all pass".into());
    }
    let mut report = Report {
        schema: SCHEMA,
        status:
            "CONTRACT_ONLY_CURRENT_CPU_COMPONENTS_CONSUMED_FUTURE_DEVICE_DISPATCH_BLOCKED_NOT_RUNTIME_NO_TOKEN_NO_HCLI_NO_TPS",
        component_contract_only: true,
        terminal_head_cpu_components: current_report(&current),
        future_device_dispatch_ledger: future_device_ledger_report(),
        decoder_readiness_integration: decoder_readiness_integration_report(),
        rejection_tests,
        rawls_hybrid_scheduler_handoff: rawls_handoff(),
        claim_boundary: vec![
            "This target read only the two explicit JSON component receipts. It did not open a Qwen80 tensor payload, safetensors shard, complete-artifact catalog, live server, watcher, HCLI adapter, or model runtime.",
            "The current terminal-head and tokenizer/sampler receipts are raw/unsealed CPU component evidence. They cannot authorize a Metal dispatch without a newly sealed wrapper binding both exact document and preimage digests.",
            "No actual all-48-layer Qwen80 hidden vector or final-head device parity receipt exists in this contract. Therefore it is NOT_RUNTIME, NO_TOKEN, NO_HCLI, NO_BASE_TRUE_TPS, NO_TG, and no capability or tournament evidence.",
        ],
        live_artifact_scan_performed: false,
        metal_device_or_dispatch_performed: false,
        model_runtime_or_server_started: false,
        hcli_execution_performed: false,
        tps_or_tg_measurement_performed: false,
        unsealed_preimage_sha256: String::new(),
    };
    report.unsealed_preimage_sha256 = sha256_hex(&serde_json::to_vec(&report)?);
    write_report_atomic(&args.out, &report)
}

fn main() {
    match parse_args().and_then(run) {
        Ok(()) => {}
        Err(error) => {
            eprintln!("ascension_qwen80_terminal_head_device_contract: {error}");
            std::process::exit(2);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn exact_direct_packed_terminal_abi_is_required() {
        PackedTensorAbi::final_norm().validate_final_norm().unwrap();
        PackedTensorAbi::lm_head().validate_lm_head().unwrap();
        assert_eq!(PackedTensorAbi::final_norm().shape, vec![2_048]);
        assert_eq!(PackedTensorAbi::lm_head().shape, vec![151_936, 2_048]);
        assert_eq!(RESERVED_TAIL_ROWS, 267);
    }

    #[test]
    fn wrong_norm_head_and_group_geometry_are_rejected() {
        let mut norm = PackedTensorAbi::final_norm();
        norm.shape = vec![HIDDEN - 1];
        assert!(norm.validate_final_norm().is_err());

        let mut head = PackedTensorAbi::lm_head();
        head.shape = vec![LM_HEAD_VOCAB, HIDDEN - 1];
        assert!(head.validate_lm_head().is_err());

        let mut grouped = PackedTensorAbi::lm_head();
        grouped.group_size = GROUP_SIZE / 2;
        assert!(grouped.validate_lm_head().is_err());
    }

    #[test]
    fn host_dispatch_requires_rms_then_all_rows_then_tail_mask_then_sample() {
        let current = current_fixture();
        let baseline = sealed_baseline_fixture(&current);
        let mut dispatch = FutureDeviceHostDispatch::begin(
            &baseline,
            &current,
            DeviceExecutionPolicy::strict_future_component(),
        )
        .unwrap();
        assert!(dispatch
            .encode_all_row_lm_head(&current.lm_head, LM_HEAD_VOCAB)
            .is_err());
        assert!(dispatch
            .mask_reserved_tail(FIRST_RESERVED_ID, RESERVED_TAIL_ROWS)
            .is_err());
        assert!(dispatch
            .deterministic_sample(DETERMINISTIC_SAMPLER, 7)
            .is_err());

        dispatch
            .bind_actual_all_layer_hidden(&all_layer_hidden_fixture())
            .unwrap();
        dispatch.encode_final_rms_norm(&current.final_norm).unwrap();
        dispatch
            .encode_all_row_lm_head(&current.lm_head, LM_HEAD_VOCAB)
            .unwrap();
        assert!(dispatch
            .deterministic_sample(DETERMINISTIC_SAMPLER, 7)
            .is_err());
        dispatch
            .mask_reserved_tail(FIRST_RESERVED_ID, RESERVED_TAIL_ROWS)
            .unwrap();
        dispatch
            .deterministic_sample(DETERMINISTIC_SAMPLER, 7)
            .unwrap();
        dispatch.validate_feedback(7).unwrap();
        assert_eq!(dispatch.finish().unwrap().len(), 6);
    }

    #[test]
    fn unsealed_or_unverified_cpu_baseline_is_rejected() {
        let current = current_fixture();
        let mut baseline = sealed_baseline_fixture(&current);
        baseline.seal_sha256 = None;
        assert!(FutureDeviceHostDispatch::begin(
            &baseline,
            &current,
            DeviceExecutionPolicy::strict_future_component(),
        )
        .is_err());

        let mut unverified = sealed_baseline_fixture(&current);
        unverified.integrity_verified = false;
        assert!(FutureDeviceHostDispatch::begin(
            &unverified,
            &current,
            DeviceExecutionPolicy::strict_future_component(),
        )
        .is_err());
    }

    #[test]
    fn source_mismatch_and_fallback_are_rejected() {
        let current = current_fixture();
        let mut mismatched = sealed_baseline_fixture(&current);
        mismatched.source.source_tokenizer_sha256 = test_sha256('c');
        assert!(FutureDeviceHostDispatch::begin(
            &mismatched,
            &current,
            DeviceExecutionPolicy::strict_future_component(),
        )
        .is_err());

        let mut fallback = DeviceExecutionPolicy::strict_future_component();
        fallback.bf16_or_decoded_weight_fallback_used = true;
        assert!(FutureDeviceHostDispatch::begin(
            &sealed_baseline_fixture(&current),
            &current,
            fallback,
        )
        .is_err());
    }

    #[test]
    fn actual_all_layer_hidden_and_device_parity_are_mandatory() {
        let current = current_fixture();
        let baseline = sealed_baseline_fixture(&current);
        let mut dispatch = FutureDeviceHostDispatch::begin(
            &baseline,
            &current,
            DeviceExecutionPolicy::strict_future_component(),
        )
        .unwrap();
        let mut hidden = all_layer_hidden_fixture();
        hidden.produced_by_all_48_layers = false;
        assert!(dispatch.bind_actual_all_layer_hidden(&hidden).is_err());

        let mut hidden = all_layer_hidden_fixture();
        hidden.device_parity_receipt_seal_sha256 = None;
        assert!(dispatch.bind_actual_all_layer_hidden(&hidden).is_err());
    }

    #[test]
    fn tail_and_feedback_domain_boundaries_are_exact() {
        let current = current_fixture();
        let baseline = sealed_baseline_fixture(&current);
        let mut dispatch = FutureDeviceHostDispatch::begin(
            &baseline,
            &current,
            DeviceExecutionPolicy::strict_future_component(),
        )
        .unwrap();
        dispatch
            .bind_actual_all_layer_hidden(&all_layer_hidden_fixture())
            .unwrap();
        dispatch.encode_final_rms_norm(&current.final_norm).unwrap();
        dispatch
            .encode_all_row_lm_head(&current.lm_head, LM_HEAD_VOCAB)
            .unwrap();
        assert!(dispatch
            .mask_reserved_tail(FIRST_RESERVED_ID - 1, RESERVED_TAIL_ROWS)
            .is_err());
        dispatch
            .mask_reserved_tail(FIRST_RESERVED_ID, RESERVED_TAIL_ROWS)
            .unwrap();
        assert!(dispatch
            .deterministic_sample(DETERMINISTIC_SAMPLER, FIRST_RESERVED_ID)
            .is_err());
    }

    #[test]
    fn rejection_report_is_complete_and_fail_closed() {
        assert!(all_rejections_pass(&rejection_report()));
        let ledger = future_device_ledger_report();
        assert!(!ledger.dispatch_authorized_now);
        assert!(!ledger.raw_component_receipts_can_authorize_device_dispatch);
        assert!(!ledger.actual_all_layer_hidden_input_available);
        assert!(!ledger.actual_device_parity_available);
    }

    #[test]
    fn decoder_readiness_mapping_stays_nonpromotable_until_full_path_device_evidence() {
        let integration = decoder_readiness_integration_report();
        assert_eq!(
            integration.target_readiness_result_schema,
            DECODER_READINESS_RESULT_SCHEMA
        );
        assert_eq!(
            integration.terminal_head_operator_classes,
            vec![
                "final_norm",
                "lm_head",
                "tail_mask",
                "sampler",
                "tokenizer_template",
            ]
        );
        assert!(!integration.current_cpu_components_are_full_path_device_ledgers);
        assert!(!integration.eligible_to_emit_promotable_readiness_descriptors_now);
    }
}

//! CPU-only Qwen3-Coder-Next 48-layer command-graph readiness contract.
//!
//! This is an isolated evidence validator, not a Qwen80 runtime. It consumes
//! compact evidence shaped like the existing complete-decoder-readiness and
//! terminal-head-device-contract reports, plus a declared physical
//! command-graph ledger. It never opens an artifact, contacts a server,
//! allocates state, creates Metal work, invokes HCLI, or measures TPS.
//!
//! A ready result is deliberately impossible unless every one of the 48
//! scheduled layers has physical, sealed, source/artifact-bound, full-path
//! device-parity evidence. The same ledger must bind per-session state slots
//! and the only legal one-token command order:
//!
//! embedding -> layers 0 through 47 -> final norm -> all-row lm_head ->
//! tail mask -> deterministic sample -> feedback.
//!
//! The built-in current-evidence mode is intentionally incomplete. It records
//! current component evidence without promoting it to a decoder token,
//! HCLI, BASE_TRUE_TPS, TG, capability, or tournament result.

use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::collections::BTreeSet;
use std::env;
use std::error::Error;
use std::fs;
use std::path::{Path, PathBuf};

const INPUT_SCHEMA: &str = "hawking.ascension.qwen80_48_layer_command_graph_readiness_input.v1";
const RESULT_SCHEMA: &str = "hawking.ascension.qwen80_48_layer_command_graph_readiness_result.v1";
const DECODER_READINESS_SCHEMA: &str =
    "hawking.ascension.qwen80_complete_decoder_readiness_result.v1";
const TERMINAL_HEAD_CONTRACT_SCHEMA: &str =
    "hawking.ascension.qwen80_terminal_head_device_contract.v1";
const TERMINAL_HEAD_LEDGER_SCHEMA: &str =
    "hawking.ascension.qwen80_terminal_head_future_device_dispatch_ledger.v1";
const PHYSICAL_GRAPH_SCHEMA: &str =
    "hawking.ascension.qwen80_48_layer_command_graph_physical_ledger.v1";
const MODEL_ID: &str = "Qwen3-Coder-Next-80B";
const MODEL_KEY: &str = "qwen80";
const SOURCE_REPOSITORY: &str = "Qwen/Qwen3-Coder-Next";
const SOURCE_REVISION: &str = "a7fbcb5c0e12d62a448eaa0e260346bf5dcc0feb";
const MANIFEST_SEAL: &str = "14cf6c4d17086dabc54b53b4dd28b9f6551ef06c6d8bf4ee8453d775d0f6817b";
const ADMISSION_RECEIPT_SEAL: &str =
    "939b41322363da3db774a2530b207bf380ed641d23cae671fc6438c0eecbf628";
const LAYER_COUNT: usize = 48;
const DELTANET_LAYERS: usize = 36;
const GQA_LAYERS: usize = 12;
const HIDDEN: usize = 2_048;
const LM_HEAD_VOCAB: usize = 151_936;
const TOKENIZER_VOCAB: usize = 151_669;
const RESERVED_TAIL_ROWS: usize = LM_HEAD_VOCAB - TOKENIZER_VOCAB;

#[derive(Clone, Copy, Debug, Deserialize, Eq, PartialEq, Serialize)]
enum Mixer {
    #[serde(rename = "deltanet")]
    DeltaNet,
    #[serde(rename = "gqa")]
    Gqa,
}

impl Mixer {
    fn expected_for_layer(layer: usize) -> Self {
        if layer % 4 == 3 {
            Self::Gqa
        } else {
            Self::DeltaNet
        }
    }

    fn as_str(self) -> &'static str {
        match self {
            Self::DeltaNet => "deltanet",
            Self::Gqa => "gqa",
        }
    }
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
struct SourceIdentity {
    model_id: String,
    model_key: String,
    source_repository: String,
    source_revision: String,
    manifest_seal_sha256: String,
    admission_receipt_seal_sha256: String,
}

impl SourceIdentity {
    fn exact() -> Self {
        Self {
            model_id: MODEL_ID.into(),
            model_key: MODEL_KEY.into(),
            source_repository: SOURCE_REPOSITORY.into(),
            source_revision: SOURCE_REVISION.into(),
            manifest_seal_sha256: MANIFEST_SEAL.into(),
            admission_receipt_seal_sha256: ADMISSION_RECEIPT_SEAL.into(),
        }
    }

    fn validate_exact(&self, label: &str) -> Result<(), String> {
        if self != &Self::exact() {
            return Err(format!("{label} source/artifact identity drifted"));
        }
        Ok(())
    }
}

#[derive(Clone, Debug, Deserialize, Serialize)]
struct OperatorCoverageEvidence {
    operator_class: String,
    satisfied: bool,
}

#[derive(Clone, Debug, Deserialize, Serialize)]
struct DecoderReadinessContractEvidence {
    schema: String,
    status: String,
    complete_decoder_readiness_earned: bool,
    source_artifact_binding: SourceIdentity,
    operator_coverage: Vec<OperatorCoverageEvidence>,
    read_only_contract: bool,
}

#[derive(Clone, Debug, Deserialize, Serialize)]
struct PackedAbiEvidence {
    shape: Vec<usize>,
    group_size: usize,
}

#[derive(Clone, Debug, Deserialize, Serialize)]
struct TerminalHeadCpuComponentsEvidence {
    source_binding: SourceIdentity,
    final_norm_abi: PackedAbiEvidence,
    lm_head_abi: PackedAbiEvidence,
    tokenizer_addressable_vocab_size: usize,
    lm_head_vocab_size: usize,
    reserved_tail_rows: usize,
    raw_component_receipts_are_unsealed: bool,
}

#[derive(Clone, Debug, Deserialize, Serialize)]
struct TerminalHeadFutureLedgerEvidence {
    schema: String,
    dispatch_authorized_now: bool,
    raw_component_receipts_can_authorize_device_dispatch: bool,
    actual_all_layer_hidden_input_available: bool,
    actual_device_parity_available: bool,
}

#[derive(Clone, Debug, Deserialize, Serialize)]
struct TerminalHeadContractEvidence {
    schema: String,
    status: String,
    component_contract_only: bool,
    terminal_head_cpu_components: TerminalHeadCpuComponentsEvidence,
    future_device_dispatch_ledger: TerminalHeadFutureLedgerEvidence,
}

#[derive(Clone, Debug, Deserialize, Serialize)]
struct StateSlotEvidence {
    layer: usize,
    slot: usize,
    source_bound: bool,
    artifact_bound: bool,
    device_resident: bool,
    full_path: bool,
    device_parity_passed: bool,
    fixture_only: bool,
    fallback_used: bool,
    receipt_seal_sha256: String,
}

#[derive(Clone, Debug, Deserialize, Serialize)]
struct PerSessionStateEvidence {
    session_id: String,
    deltanet_slots: Vec<StateSlotEvidence>,
    gqa_slots: Vec<StateSlotEvidence>,
}

#[derive(Clone, Debug, Deserialize, Serialize)]
struct LayerPhysicalEvidence {
    layer: usize,
    mixer: Mixer,
    receipt_seal_sha256: String,
    backend: String,
    source_bound: bool,
    artifact_bound: bool,
    full_path: bool,
    complete_token_path: bool,
    device_parity_passed: bool,
    fixture_only: bool,
    synthetic_input: bool,
    component_only: bool,
    fallback_used: bool,
}

#[derive(Clone, Debug, Deserialize, Serialize)]
struct TerminalPhysicalEvidence {
    receipt_seal_sha256: String,
    backend: String,
    source_bound: bool,
    artifact_bound: bool,
    full_path: bool,
    complete_token_path: bool,
    device_parity_passed: bool,
    fixture_only: bool,
    synthetic_input: bool,
    component_only: bool,
    fallback_used: bool,
    final_norm: bool,
    all_row_lm_head: bool,
    tail_mask_before_sample: bool,
    deterministic_sample: bool,
    tokenizer_addressable_feedback: bool,
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
enum CommandStep {
    Embedding,
    Layer { layer: usize },
    FinalNorm,
    AllRowLmHead,
    TailMask,
    DeterministicSample,
    Feedback,
}

impl CommandStep {
    fn display(&self) -> String {
        match self {
            Self::Embedding => "embedding".into(),
            Self::Layer { layer } => format!("layer-{layer}"),
            Self::FinalNorm => "final_norm".into(),
            Self::AllRowLmHead => "all_row_lm_head".into(),
            Self::TailMask => "tail_mask".into(),
            Self::DeterministicSample => "deterministic_sample".into(),
            Self::Feedback => "feedback".into(),
        }
    }
}

#[derive(Clone, Debug, Deserialize, Serialize)]
struct PhysicalCommandGraphEvidence {
    schema: String,
    graph_receipt_seal_sha256: String,
    source_identity: SourceIdentity,
    physical_capture: bool,
    session_state: PerSessionStateEvidence,
    layers: Vec<LayerPhysicalEvidence>,
    terminal: TerminalPhysicalEvidence,
    command_steps: Vec<CommandStep>,
}

#[derive(Clone, Debug, Deserialize, Serialize)]
struct ReadinessInput {
    schema: String,
    decoder_readiness_contract: DecoderReadinessContractEvidence,
    terminal_head_contract: TerminalHeadContractEvidence,
    physical_command_graph: PhysicalCommandGraphEvidence,
}

#[derive(Serialize)]
struct LayerCoverageReport {
    layer: usize,
    expected_mixer: &'static str,
    evidence_present: bool,
    physical_full_path_device_evidence_valid: bool,
    reasons: Vec<String>,
}

#[derive(Serialize)]
struct StateSlotReport {
    session_id: String,
    expected_deltanet_slots: usize,
    observed_deltanet_slots: usize,
    expected_gqa_slots: usize,
    observed_gqa_slots: usize,
    valid: bool,
    reasons: Vec<String>,
}

#[derive(Serialize)]
struct Report {
    schema: &'static str,
    status: &'static str,
    command_graph_ready: bool,
    decoder_readiness_contract_schema_valid: bool,
    terminal_head_contract_schema_valid: bool,
    exact_48_layer_schedule_valid: bool,
    per_session_state_slots_valid: bool,
    physical_full_path_device_evidence_valid_for_every_layer: bool,
    terminal_physical_path_valid: bool,
    command_order_valid: bool,
    decoder_contract_is_context_not_circular_promotion: bool,
    source_identity_valid: bool,
    graph_receipt_sealed: bool,
    layer_coverage: Vec<LayerCoverageReport>,
    state_slots: StateSlotReport,
    required_command_order: Vec<String>,
    missing_or_invalid_layers: Vec<usize>,
    blockers: Vec<String>,
    current_component_evidence_is_incomplete: bool,
    read_only_contract: bool,
    live_artifact_scan_performed: bool,
    metal_device_or_dispatch_performed: bool,
    model_execution_performed: bool,
    hcli_execution_performed: bool,
    tps_or_tg_measurement_performed: bool,
    claim_boundary: Vec<&'static str>,
    unsealed_preimage_sha256: String,
}

enum InputMode {
    Input(PathBuf),
    CurrentEvidence,
}

struct Args {
    input_mode: InputMode,
    out: PathBuf,
}

fn is_lower_sha256(value: &str) -> bool {
    value.len() == 64
        && value
            .bytes()
            .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte))
}

fn sha256_hex(bytes: &[u8]) -> String {
    format!("{:x}", Sha256::digest(bytes))
}

fn expected_schedule() -> Vec<Mixer> {
    (0..LAYER_COUNT).map(Mixer::expected_for_layer).collect()
}

fn deltanet_layer_numbers() -> Vec<usize> {
    (0..LAYER_COUNT)
        .filter(|layer| Mixer::expected_for_layer(*layer) == Mixer::DeltaNet)
        .collect()
}

fn gqa_layer_numbers() -> Vec<usize> {
    (0..LAYER_COUNT)
        .filter(|layer| Mixer::expected_for_layer(*layer) == Mixer::Gqa)
        .collect()
}

fn required_command_steps() -> Vec<CommandStep> {
    let mut steps = Vec::with_capacity(LAYER_COUNT + 6);
    steps.push(CommandStep::Embedding);
    steps.extend((0..LAYER_COUNT).map(|layer| CommandStep::Layer { layer }));
    steps.push(CommandStep::FinalNorm);
    steps.push(CommandStep::AllRowLmHead);
    steps.push(CommandStep::TailMask);
    steps.push(CommandStep::DeterministicSample);
    steps.push(CommandStep::Feedback);
    steps
}

fn expected_operator_classes() -> [&'static str; 7] {
    [
        "command_graph_48_layer",
        "final_norm",
        "lm_head",
        "tail_mask",
        "sampler",
        "tokenizer_template",
        "hcli_adapter",
    ]
}

fn validate_decoder_contract(
    evidence: &DecoderReadinessContractEvidence,
    blockers: &mut Vec<String>,
) -> bool {
    let mut valid = true;
    if evidence.schema != DECODER_READINESS_SCHEMA {
        blockers.push("decoder-readiness evidence schema drifted".into());
        valid = false;
    }
    if evidence.status.trim().is_empty() {
        blockers.push("decoder-readiness evidence status is empty".into());
        valid = false;
    }
    if !evidence.read_only_contract {
        blockers.push("decoder-readiness evidence must remain a read-only contract".into());
        valid = false;
    }
    if let Err(error) = evidence
        .source_artifact_binding
        .validate_exact("decoder-readiness evidence")
    {
        blockers.push(error);
        valid = false;
    }
    let classes = evidence
        .operator_coverage
        .iter()
        .map(|entry| entry.operator_class.as_str())
        .collect::<BTreeSet<_>>();
    for required in expected_operator_classes() {
        if !classes.contains(required) {
            blockers.push(format!(
                "decoder-readiness evidence omits required operator class {required}"
            ));
            valid = false;
        }
    }
    if evidence.complete_decoder_readiness_earned
        && evidence
            .operator_coverage
            .iter()
            .any(|entry| !entry.satisfied)
    {
        blockers.push(
            "decoder-readiness evidence claims earned while an operator coverage entry is unsatisfied"
                .into(),
        );
        valid = false;
    }
    valid
}

fn validate_terminal_contract(
    evidence: &TerminalHeadContractEvidence,
    expected_identity: &SourceIdentity,
    blockers: &mut Vec<String>,
) -> bool {
    let mut valid = true;
    if evidence.schema != TERMINAL_HEAD_CONTRACT_SCHEMA {
        blockers.push("terminal-head contract evidence schema drifted".into());
        valid = false;
    }
    if evidence.status.trim().is_empty() {
        blockers.push("terminal-head contract evidence status is empty".into());
        valid = false;
    }
    if evidence.future_device_dispatch_ledger.schema != TERMINAL_HEAD_LEDGER_SCHEMA {
        blockers.push("terminal-head future-device ledger schema drifted".into());
        valid = false;
    }
    if evidence.component_contract_only
        && evidence
            .future_device_dispatch_ledger
            .raw_component_receipts_can_authorize_device_dispatch
    {
        blockers.push(
            "terminal-head contract cannot mark raw component receipts as device-dispatch authority"
                .into(),
        );
        valid = false;
    }
    if evidence.terminal_head_cpu_components.source_binding != *expected_identity {
        blockers
            .push("terminal-head CPU component source binding differs from decoder binding".into());
        valid = false;
    }
    if evidence
        .terminal_head_cpu_components
        .final_norm_abi
        .shape
        .as_slice()
        != [HIDDEN]
        || evidence
            .terminal_head_cpu_components
            .lm_head_abi
            .shape
            .as_slice()
            != [LM_HEAD_VOCAB, HIDDEN]
        || evidence
            .terminal_head_cpu_components
            .final_norm_abi
            .group_size
            != 128
        || evidence.terminal_head_cpu_components.lm_head_abi.group_size != 128
        || evidence
            .terminal_head_cpu_components
            .tokenizer_addressable_vocab_size
            != TOKENIZER_VOCAB
        || evidence.terminal_head_cpu_components.lm_head_vocab_size != LM_HEAD_VOCAB
        || evidence.terminal_head_cpu_components.reserved_tail_rows != RESERVED_TAIL_ROWS
    {
        blockers.push("terminal-head CPU ABI/tail geometry drifted".into());
        valid = false;
    }
    valid
}

fn validate_state_slots(state: &PerSessionStateEvidence) -> StateSlotReport {
    let mut reasons = Vec::new();
    if state.session_id.trim().is_empty() {
        reasons.push("session_id must be non-empty".into());
    }
    let validate_domain = |slots: &[StateSlotEvidence],
                           expected_layers: &[usize],
                           label: &str,
                           reasons: &mut Vec<String>| {
        if slots.len() != expected_layers.len() {
            reasons.push(format!(
                "{label} slot count {} does not equal expected {}",
                slots.len(),
                expected_layers.len()
            ));
        }
        let mut seen_layers = BTreeSet::new();
        let mut seen_slots = BTreeSet::new();
        for (expected_slot, expected_layer) in expected_layers.iter().copied().enumerate() {
            let Some(slot) = slots.iter().find(|slot| slot.layer == expected_layer) else {
                reasons.push(format!("{label} layer {expected_layer} has no state slot"));
                continue;
            };
            if slot.slot != expected_slot {
                reasons.push(format!(
                    "{label} layer {expected_layer} has slot {}, expected {expected_slot}",
                    slot.slot
                ));
            }
            if !seen_layers.insert(slot.layer) || !seen_slots.insert(slot.slot) {
                reasons.push(format!("{label} reuses layer or state slot {}", slot.slot));
            }
            if !slot.source_bound
                || !slot.artifact_bound
                || !slot.device_resident
                || !slot.full_path
                || !slot.device_parity_passed
                || slot.fixture_only
                || slot.fallback_used
                || !is_lower_sha256(&slot.receipt_seal_sha256)
            {
                reasons.push(format!(
                    "{label} layer {expected_layer} lacks physical full-path device state evidence"
                ));
            }
        }
        for slot in slots {
            if !expected_layers.contains(&slot.layer) {
                reasons.push(format!("{label} carries unexpected layer {}", slot.layer));
            }
        }
    };
    validate_domain(
        &state.deltanet_slots,
        &deltanet_layer_numbers(),
        "deltanet",
        &mut reasons,
    );
    validate_domain(&state.gqa_slots, &gqa_layer_numbers(), "gqa", &mut reasons);
    StateSlotReport {
        session_id: state.session_id.clone(),
        expected_deltanet_slots: DELTANET_LAYERS,
        observed_deltanet_slots: state.deltanet_slots.len(),
        expected_gqa_slots: GQA_LAYERS,
        observed_gqa_slots: state.gqa_slots.len(),
        valid: reasons.is_empty(),
        reasons,
    }
}

fn assess_layers(layers: &[LayerPhysicalEvidence]) -> (Vec<LayerCoverageReport>, Vec<usize>, bool) {
    let mut reports = Vec::with_capacity(LAYER_COUNT);
    let mut missing_or_invalid = Vec::new();
    let mut duplicate_layers = BTreeSet::new();
    let mut seen = BTreeSet::new();
    for layer in layers {
        if !seen.insert(layer.layer) {
            duplicate_layers.insert(layer.layer);
        }
    }
    for layer in 0..LAYER_COUNT {
        let mut reasons = Vec::new();
        let matching = layers
            .iter()
            .filter(|entry| entry.layer == layer)
            .collect::<Vec<_>>();
        if matching.is_empty() {
            reasons.push("no physical layer evidence".into());
        } else if matching.len() != 1 || duplicate_layers.contains(&layer) {
            reasons.push("duplicate physical layer evidence".into());
        } else {
            let evidence = matching[0];
            if evidence.mixer != Mixer::expected_for_layer(layer) {
                reasons.push(format!(
                    "mixer is {}, expected {}",
                    evidence.mixer.as_str(),
                    Mixer::expected_for_layer(layer).as_str()
                ));
            }
            if evidence.backend != "metal"
                || !evidence.source_bound
                || !evidence.artifact_bound
                || !evidence.full_path
                || !evidence.complete_token_path
                || !evidence.device_parity_passed
                || evidence.fixture_only
                || evidence.synthetic_input
                || evidence.component_only
                || evidence.fallback_used
                || !is_lower_sha256(&evidence.receipt_seal_sha256)
            {
                reasons.push(
                    "requires sealed strict-Metal source/artifact-bound full-path complete-token parity without fixture, component, synthetic, or fallback execution"
                        .into(),
                );
            }
        }
        let valid = reasons.is_empty();
        if !valid {
            missing_or_invalid.push(layer);
        }
        reports.push(LayerCoverageReport {
            layer,
            expected_mixer: Mixer::expected_for_layer(layer).as_str(),
            evidence_present: !matching.is_empty(),
            physical_full_path_device_evidence_valid: valid,
            reasons,
        });
    }
    (
        reports,
        missing_or_invalid.clone(),
        missing_or_invalid.is_empty(),
    )
}

fn exact_schedule_from_layer_evidence(layers: &[LayerPhysicalEvidence]) -> bool {
    if layers.len() != LAYER_COUNT {
        return false;
    }
    let mut seen = BTreeSet::new();
    for evidence in layers {
        if evidence.layer >= LAYER_COUNT
            || !seen.insert(evidence.layer)
            || evidence.mixer != Mixer::expected_for_layer(evidence.layer)
        {
            return false;
        }
    }
    seen.len() == LAYER_COUNT
}

fn validate_terminal_physical(
    terminal: &TerminalPhysicalEvidence,
    blockers: &mut Vec<String>,
) -> bool {
    let valid = terminal.backend == "metal"
        && terminal.source_bound
        && terminal.artifact_bound
        && terminal.full_path
        && terminal.complete_token_path
        && terminal.device_parity_passed
        && !terminal.fixture_only
        && !terminal.synthetic_input
        && !terminal.component_only
        && !terminal.fallback_used
        && terminal.final_norm
        && terminal.all_row_lm_head
        && terminal.tail_mask_before_sample
        && terminal.deterministic_sample
        && terminal.tokenizer_addressable_feedback
        && is_lower_sha256(&terminal.receipt_seal_sha256);
    if !valid {
        blockers.push(
            "terminal physical evidence must be sealed strict-Metal full-path parity for final norm, all-row lm_head, tail mask before sample, deterministic sample, and feedback"
                .into(),
        );
    }
    valid
}

fn validate_command_order(steps: &[CommandStep], blockers: &mut Vec<String>) -> bool {
    let expected = required_command_steps();
    if steps != expected {
        blockers.push(format!(
            "command order drifted: expected [{}], observed [{}]",
            expected
                .iter()
                .map(CommandStep::display)
                .collect::<Vec<_>>()
                .join(", "),
            steps
                .iter()
                .map(CommandStep::display)
                .collect::<Vec<_>>()
                .join(", "),
        ));
        false
    } else {
        true
    }
}

fn evaluate(input: ReadinessInput) -> Report {
    let mut blockers = Vec::new();
    let input_schema_valid = input.schema == INPUT_SCHEMA;
    if !input_schema_valid {
        blockers.push("input schema drifted".into());
    }
    let decoder_readiness_contract_schema_valid =
        validate_decoder_contract(&input.decoder_readiness_contract, &mut blockers);
    let source_identity_valid = input
        .physical_command_graph
        .source_identity
        .validate_exact("physical command graph")
        .is_ok()
        && input.physical_command_graph.source_identity
            == input.decoder_readiness_contract.source_artifact_binding;
    if !source_identity_valid {
        blockers
            .push("physical command graph source identity differs from decoder evidence".into());
    }
    let terminal_head_contract_schema_valid = validate_terminal_contract(
        &input.terminal_head_contract,
        &input.decoder_readiness_contract.source_artifact_binding,
        &mut blockers,
    );
    let physical_graph_schema_valid = input.physical_command_graph.schema == PHYSICAL_GRAPH_SCHEMA;
    if !physical_graph_schema_valid {
        blockers.push("physical command-graph ledger schema drifted".into());
    }
    let (layer_coverage, missing_or_invalid_layers, physical_layers_valid) =
        assess_layers(&input.physical_command_graph.layers);
    if !physical_layers_valid {
        blockers.push(format!(
            "physical full-path/device evidence is incomplete for layers {:?}",
            missing_or_invalid_layers
        ));
    }
    let state_slots = validate_state_slots(&input.physical_command_graph.session_state);
    if !state_slots.valid {
        blockers.push("per-session 36 DeltaNet + 12 GQA state-slot evidence is incomplete".into());
    }
    let terminal_physical_path_valid =
        validate_terminal_physical(&input.physical_command_graph.terminal, &mut blockers);
    let command_order_valid =
        validate_command_order(&input.physical_command_graph.command_steps, &mut blockers);
    let graph_receipt_sealed =
        is_lower_sha256(&input.physical_command_graph.graph_receipt_seal_sha256);
    if !graph_receipt_sealed {
        blockers.push("command graph physical receipt must carry a lowercase SHA-256 seal".into());
    }
    if !input.physical_command_graph.physical_capture {
        blockers.push("command graph evidence is not marked as a physical capture".into());
    }

    // The pre-existing decoder readiness report is a requirements inventory.
    // Requiring its global ready bit here would be circular because it itself
    // correctly waits for a command_graph_48_layer descriptor.
    let decoder_contract_is_context_not_circular_promotion = true;
    let exact_48_layer_schedule_valid =
        exact_schedule_from_layer_evidence(&input.physical_command_graph.layers);
    let command_graph_ready = input_schema_valid
        && decoder_readiness_contract_schema_valid
        && terminal_head_contract_schema_valid
        && physical_graph_schema_valid
        && source_identity_valid
        && exact_48_layer_schedule_valid
        && state_slots.valid
        && physical_layers_valid
        && terminal_physical_path_valid
        && command_order_valid
        && graph_receipt_sealed
        && input.physical_command_graph.physical_capture;
    let status = if command_graph_ready {
        "READY_QWEN80_48_LAYER_COMMAND_GRAPH_PHYSICAL_FULL_PATH_EVIDENCE_CONTRACT_ONLY"
    } else {
        "INCOMPLETE_QWEN80_48_LAYER_COMMAND_GRAPH_NOT_RUNTIME_NO_TOKEN_NO_HCLI_NO_TPS"
    };
    let current_component_evidence_is_incomplete =
        input.terminal_head_contract.component_contract_only
            || !input
                .terminal_head_contract
                .future_device_dispatch_ledger
                .dispatch_authorized_now
            || !input
                .terminal_head_contract
                .future_device_dispatch_ledger
                .actual_all_layer_hidden_input_available
            || !input
                .terminal_head_contract
                .future_device_dispatch_ledger
                .actual_device_parity_available
            || input
                .terminal_head_contract
                .terminal_head_cpu_components
                .raw_component_receipts_are_unsealed
            || !input
                .decoder_readiness_contract
                .complete_decoder_readiness_earned;
    let mut report = Report {
        schema: RESULT_SCHEMA,
        status,
        command_graph_ready,
        decoder_readiness_contract_schema_valid,
        terminal_head_contract_schema_valid,
        exact_48_layer_schedule_valid,
        per_session_state_slots_valid: state_slots.valid,
        physical_full_path_device_evidence_valid_for_every_layer: physical_layers_valid,
        terminal_physical_path_valid,
        command_order_valid,
        decoder_contract_is_context_not_circular_promotion,
        source_identity_valid,
        graph_receipt_sealed,
        layer_coverage,
        state_slots,
        required_command_order: required_command_steps()
            .iter()
            .map(CommandStep::display)
            .collect(),
        missing_or_invalid_layers,
        blockers,
        current_component_evidence_is_incomplete,
        read_only_contract: true,
        live_artifact_scan_performed: false,
        metal_device_or_dispatch_performed: false,
        model_execution_performed: false,
        hcli_execution_performed: false,
        tps_or_tg_measurement_performed: false,
        claim_boundary: vec![
            "This target consumes only supplied JSON evidence descriptors. It does not open a Qwen80 artifact, source shard, tensor payload, state buffer, server, watcher, HCLI endpoint, or Metal device.",
            "Current component evidence is deliberately incomplete. A ready result requires separately sealed physical full-path device evidence for all 48 layers, session state, terminal path, and command order.",
            "Even a ready contract is command-graph evidence only. It is not a model execution, generated token, HCLI result, BASE_TRUE_TPS/TG measurement, capability result, or tournament action.",
        ],
        unsealed_preimage_sha256: String::new(),
    };
    report.unsealed_preimage_sha256 =
        sha256_hex(&serde_json::to_vec(&report).expect("report serializes"));
    report
}

fn test_sha256(fill: char) -> String {
    std::iter::repeat_n(fill, 64).collect()
}

fn expected_state_slots(layers: &[usize], seal: &str, physical: bool) -> Vec<StateSlotEvidence> {
    layers
        .iter()
        .copied()
        .enumerate()
        .map(|(slot, layer)| StateSlotEvidence {
            layer,
            slot,
            source_bound: physical,
            artifact_bound: physical,
            device_resident: physical,
            full_path: physical,
            device_parity_passed: physical,
            fixture_only: !physical,
            fallback_used: false,
            receipt_seal_sha256: seal.into(),
        })
        .collect()
}

fn expected_layer_evidence(physical: bool) -> Vec<LayerPhysicalEvidence> {
    (0..LAYER_COUNT)
        .map(|layer| LayerPhysicalEvidence {
            layer,
            mixer: Mixer::expected_for_layer(layer),
            receipt_seal_sha256: test_sha256('c'),
            backend: if physical { "metal" } else { "cpu" }.into(),
            source_bound: physical,
            artifact_bound: physical,
            full_path: physical,
            complete_token_path: physical,
            device_parity_passed: physical,
            fixture_only: !physical,
            synthetic_input: !physical,
            component_only: !physical,
            fallback_used: false,
        })
        .collect()
}

fn decoder_contract_fixture() -> DecoderReadinessContractEvidence {
    DecoderReadinessContractEvidence {
        schema: DECODER_READINESS_SCHEMA.into(),
        status: "INCOMPLETE_QWEN80_COMPLETE_DECODER_READINESS_NO_DECODER_TOKEN_HCLI_OR_TPS_CLAIM"
            .into(),
        complete_decoder_readiness_earned: false,
        source_artifact_binding: SourceIdentity::exact(),
        operator_coverage: expected_operator_classes()
            .iter()
            .map(|operator_class| OperatorCoverageEvidence {
                operator_class: (*operator_class).into(),
                satisfied: false,
            })
            .collect(),
        read_only_contract: true,
    }
}

fn terminal_contract_fixture() -> TerminalHeadContractEvidence {
    TerminalHeadContractEvidence {
        schema: TERMINAL_HEAD_CONTRACT_SCHEMA.into(),
        status:
            "CONTRACT_ONLY_CURRENT_CPU_COMPONENTS_CONSUMED_FUTURE_DEVICE_DISPATCH_BLOCKED_NOT_RUNTIME_NO_TOKEN_NO_HCLI_NO_TPS"
                .into(),
        component_contract_only: true,
        terminal_head_cpu_components: TerminalHeadCpuComponentsEvidence {
            source_binding: SourceIdentity::exact(),
            final_norm_abi: PackedAbiEvidence {
                shape: vec![HIDDEN],
                group_size: 128,
            },
            lm_head_abi: PackedAbiEvidence {
                shape: vec![LM_HEAD_VOCAB, HIDDEN],
                group_size: 128,
            },
            tokenizer_addressable_vocab_size: TOKENIZER_VOCAB,
            lm_head_vocab_size: LM_HEAD_VOCAB,
            reserved_tail_rows: RESERVED_TAIL_ROWS,
            raw_component_receipts_are_unsealed: true,
        },
        future_device_dispatch_ledger: TerminalHeadFutureLedgerEvidence {
            schema: TERMINAL_HEAD_LEDGER_SCHEMA.into(),
            dispatch_authorized_now: false,
            raw_component_receipts_can_authorize_device_dispatch: false,
            actual_all_layer_hidden_input_available: false,
            actual_device_parity_available: false,
        },
    }
}

fn terminal_physical_fixture(physical: bool) -> TerminalPhysicalEvidence {
    TerminalPhysicalEvidence {
        receipt_seal_sha256: test_sha256('d'),
        backend: if physical { "metal" } else { "cpu" }.into(),
        source_bound: physical,
        artifact_bound: physical,
        full_path: physical,
        complete_token_path: physical,
        device_parity_passed: physical,
        fixture_only: !physical,
        synthetic_input: !physical,
        component_only: !physical,
        fallback_used: false,
        final_norm: physical,
        all_row_lm_head: physical,
        tail_mask_before_sample: physical,
        deterministic_sample: physical,
        tokenizer_addressable_feedback: physical,
    }
}

fn complete_input_fixture() -> ReadinessInput {
    ReadinessInput {
        schema: INPUT_SCHEMA.into(),
        decoder_readiness_contract: decoder_contract_fixture(),
        terminal_head_contract: terminal_contract_fixture(),
        physical_command_graph: PhysicalCommandGraphEvidence {
            schema: PHYSICAL_GRAPH_SCHEMA.into(),
            graph_receipt_seal_sha256: test_sha256('e'),
            source_identity: SourceIdentity::exact(),
            physical_capture: true,
            session_state: PerSessionStateEvidence {
                session_id: "session-physical-contract-fixture".into(),
                deltanet_slots: expected_state_slots(
                    &deltanet_layer_numbers(),
                    &test_sha256('f'),
                    true,
                ),
                gqa_slots: expected_state_slots(&gqa_layer_numbers(), &test_sha256('f'), true),
            },
            layers: expected_layer_evidence(true),
            terminal: terminal_physical_fixture(true),
            command_steps: required_command_steps(),
        },
    }
}

fn current_evidence_input() -> ReadinessInput {
    let mut input = complete_input_fixture();
    input.physical_command_graph.physical_capture = false;
    input.physical_command_graph.layers = expected_layer_evidence(false);
    input.physical_command_graph.layers.truncate(1);
    input.physical_command_graph.session_state.deltanet_slots =
        expected_state_slots(&deltanet_layer_numbers(), &test_sha256('f'), false);
    input.physical_command_graph.session_state.gqa_slots =
        expected_state_slots(&gqa_layer_numbers(), &test_sha256('f'), false);
    input.physical_command_graph.terminal = terminal_physical_fixture(false);
    input
}

fn read_input(path: &Path) -> Result<ReadinessInput, Box<dyn Error>> {
    if !path.is_absolute() {
        return Err("--input must be an absolute path".into());
    }
    let metadata = fs::symlink_metadata(path)?;
    if !metadata.file_type().is_file() || metadata.file_type().is_symlink() {
        return Err("--input must be a regular non-symlink JSON file".into());
    }
    let bytes = fs::read(path)?;
    Ok(serde_json::from_slice(&bytes)?)
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
    "usage: ascension_qwen80_48_layer_command_graph_readiness_contract \
--current-evidence --out ABSOLUTE_PATH | --input ABSOLUTE_PATH --out ABSOLUTE_PATH"
}

fn parse_args() -> Result<Args, Box<dyn Error>> {
    let mut input = None;
    let mut current_evidence = false;
    let mut out = None;
    let mut values = env::args().skip(1);
    while let Some(flag) = values.next() {
        match flag.as_str() {
            "--input" => {
                let value = values.next().ok_or("missing path after --input")?;
                if input.replace(PathBuf::from(value)).is_some() {
                    return Err("--input repeated".into());
                }
            }
            "--current-evidence" => {
                if current_evidence {
                    return Err("--current-evidence repeated".into());
                }
                current_evidence = true;
            }
            "--out" => {
                let value = values.next().ok_or("missing path after --out")?;
                if out.replace(PathBuf::from(value)).is_some() {
                    return Err("--out repeated".into());
                }
            }
            _ => return Err(format!("unsupported option {flag:?}; {}", usage()).into()),
        }
    }
    let input_mode = match (input, current_evidence) {
        (Some(path), false) => InputMode::Input(path),
        (None, true) => InputMode::CurrentEvidence,
        _ => return Err(usage().into()),
    };
    let out = out.ok_or(usage())?;
    if !out.is_absolute() {
        return Err("--out must be an absolute path".into());
    }
    Ok(Args { input_mode, out })
}

fn run(args: Args) -> Result<(), Box<dyn Error>> {
    let input = match args.input_mode {
        InputMode::Input(path) => read_input(&path)?,
        InputMode::CurrentEvidence => current_evidence_input(),
    };
    let report = evaluate(input);
    write_report_atomic(&args.out, &report)
}

fn main() {
    match parse_args().and_then(run) {
        Ok(()) => {}
        Err(error) => {
            eprintln!("ascension_qwen80_48_layer_command_graph_readiness_contract: {error}");
            std::process::exit(2);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn exact_schedule_has_36_deltanet_and_12_gqa_layers() {
        let schedule = expected_schedule();
        assert_eq!(schedule.len(), LAYER_COUNT);
        assert_eq!(
            schedule
                .iter()
                .filter(|mixer| **mixer == Mixer::DeltaNet)
                .count(),
            DELTANET_LAYERS
        );
        assert_eq!(
            schedule
                .iter()
                .filter(|mixer| **mixer == Mixer::Gqa)
                .count(),
            GQA_LAYERS
        );
        assert_eq!(schedule[0], Mixer::DeltaNet);
        assert_eq!(schedule[3], Mixer::Gqa);
        assert_eq!(schedule[47], Mixer::Gqa);
    }

    #[test]
    fn current_component_evidence_is_incomplete_without_physical_layers() {
        let report = evaluate(current_evidence_input());
        assert!(!report.command_graph_ready);
        assert!(report.current_component_evidence_is_incomplete);
        assert!(!report.exact_48_layer_schedule_valid);
        assert!(!report.physical_full_path_device_evidence_valid_for_every_layer);
        assert!(!report.terminal_physical_path_valid);
        assert_eq!(report.missing_or_invalid_layers.len(), LAYER_COUNT);
        assert!(report
            .blockers
            .iter()
            .any(|blocker| blocker.contains("physical full-path/device evidence")));
    }

    #[test]
    fn complete_declared_physical_ledger_satisfies_contract_semantics() {
        let report = evaluate(complete_input_fixture());
        assert!(report.command_graph_ready);
        assert!(report.exact_48_layer_schedule_valid);
        assert!(report.per_session_state_slots_valid);
        assert!(report.physical_full_path_device_evidence_valid_for_every_layer);
        assert!(report.terminal_physical_path_valid);
        assert!(report.command_order_valid);
        assert!(report.blockers.is_empty());
    }

    #[test]
    fn wrong_hybrid_schedule_is_rejected_layer_by_layer() {
        let mut input = complete_input_fixture();
        input.physical_command_graph.layers[3].mixer = Mixer::DeltaNet;
        let report = evaluate(input);
        assert!(!report.command_graph_ready);
        assert_eq!(report.missing_or_invalid_layers, vec![3]);
        assert!(report.layer_coverage[3]
            .reasons
            .iter()
            .any(|reason| reason.contains("expected gqa")));
    }

    #[test]
    fn partial_or_component_only_layer_evidence_cannot_be_ready() {
        let mut input = complete_input_fixture();
        input.physical_command_graph.layers[17].component_only = true;
        input.physical_command_graph.layers[17].complete_token_path = false;
        let report = evaluate(input);
        assert!(!report.command_graph_ready);
        assert_eq!(report.missing_or_invalid_layers, vec![17]);
        assert!(!report.layer_coverage[17].physical_full_path_device_evidence_valid);
    }

    #[test]
    fn state_slot_alias_and_missing_device_state_are_rejected() {
        let mut input = complete_input_fixture();
        input.physical_command_graph.session_state.deltanet_slots[2].slot = 0;
        input.physical_command_graph.session_state.gqa_slots[0].device_resident = false;
        let report = evaluate(input);
        assert!(!report.command_graph_ready);
        assert!(!report.per_session_state_slots_valid);
        assert!(report
            .state_slots
            .reasons
            .iter()
            .any(|reason| reason.contains("reuses layer or state slot")));
    }

    #[test]
    fn feedback_before_tail_mask_is_rejected_by_exact_command_order() {
        let mut input = complete_input_fixture();
        let final_index = input
            .physical_command_graph
            .command_steps
            .iter()
            .position(|step| *step == CommandStep::TailMask)
            .unwrap();
        input
            .physical_command_graph
            .command_steps
            .swap(final_index, final_index + 2);
        let report = evaluate(input);
        assert!(!report.command_graph_ready);
        assert!(!report.command_order_valid);
        assert!(report
            .blockers
            .iter()
            .any(|blocker| blocker.contains("command order drifted")));
    }

    #[test]
    fn contract_schema_and_terminal_abi_source_drift_are_rejected() {
        let mut input = complete_input_fixture();
        input
            .terminal_head_contract
            .terminal_head_cpu_components
            .lm_head_abi
            .shape = vec![LM_HEAD_VOCAB - 1, HIDDEN];
        input.decoder_readiness_contract.schema = "wrong.schema".into();
        let report = evaluate(input);
        assert!(!report.command_graph_ready);
        assert!(!report.decoder_readiness_contract_schema_valid);
        assert!(!report.terminal_head_contract_schema_valid);
    }
}

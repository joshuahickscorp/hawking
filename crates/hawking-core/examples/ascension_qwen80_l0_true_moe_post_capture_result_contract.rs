//! CPU-only post-capture result contract for the Qwen80 L0 true-MoE graph.
//!
//! This example is intentionally a validator for a *future* strict component
//! capture.  It neither opens any receipt/artifact path nor invokes Metal. A
//! valid result must join one immutable fixed-ABI plan, the first-residual
//! bridge outer/inner pair, and the true-input graph outer/inner pair using
//! one real source input, retained first-residual hash, and token command
//! buffer hash.  A historical materialized or synthetic fixture is refused.

use serde::Serialize;
use serde_json::json;
use std::env;

const SCHEMA: &str = "hawking.ascension.qwen80_l0_true_moe_post_capture_result_contract.v1";
const STATUS: &str = "PREPARED_QWEN80_L0_TRUE_MOE_POST_CAPTURE_RESULT_CONTRACT_INCOMPLETE";
const CURRENT_EVIDENCE_STATE: &str = "PREPARED_INCOMPLETE_FUTURE_CAPTURE_EVIDENCE_NOT_PRESENT";
const COMPONENT_ONLY_RESULT: &str =
    "VALIDATED_QWEN80_L0_TRUE_MOE_COMPONENT_CAPTURE_NOT_COMPLETE_LAYER_OR_TOKEN";

const FIXED_ABI_SCHEMA: &str = "hawking.ascension.qwen80_l0_true_moe_fixed_payload_contract.v1";
const FIXED_ABI_STATUS: &str = "PREPARED_QWEN80_L0_TRUE_MOE_FIXED_SUFFIX_PAYLOAD_PLAN_NOT_EXECUTED";
const FIRST_RESIDUAL_OUTER_SCHEMA: &str =
    "hawking.ascension.qwen80_first_residual_outer_capture.v1";
const FIRST_RESIDUAL_OUTER_STATUS: &str =
    "CAPTURED_QWEN80_FIRST_RESIDUAL_STRICT_MATH_COMPONENT_ONLY";
const FIRST_RESIDUAL_INNER_SCHEMA: &str =
    "hawking.ascension.qwen80_first_residual_bridge_device.v1";
const FIRST_RESIDUAL_INNER_STATUS: &str =
    "EARNED_CURRENT_ADMITTED_QWEN80_DIRECT_PACKED_MIXER_FIRST_RESIDUAL_STRICT_MATH_METAL_COMPONENT_NOT_COMPLETE_LAYER_OR_TOKEN";
const TRUE_GRAPH_OUTER_SCHEMA: &str =
    "hawking.ascension.qwen80_true_input_all_ten_moe_graph_outer_launcher.v1";
const TRUE_GRAPH_OUTER_STATUS: &str =
    "CAPTURED_QWEN80_TRUE_INPUT_ALL_TEN_MOE_GRAPH_OUTER_TERMINAL_COMPONENT_ONLY";
const TRUE_GRAPH_INNER_SCHEMA: &str = "hawking.ascension.qwen80_all_ten_true_moe_graph_device.v1";
const TRUE_GRAPH_INNER_STATUS: &str =
    "EARNED_CURRENT_ADMITTED_QWEN80_LAYER0_TRUE_INPUT_ALL_TEN_ROUTE_SHARED_SECOND_RESIDUAL_STRICT_MATH_METAL_COMPONENT_NOT_COMPLETE_LAYER_OR_TOKEN";

const HIDDEN: usize = 2_048;
const TOP_K: usize = 10;
const PREFIX_DISPATCH_COUNT: usize = 9;
const SUFFIX_DISPATCH_COUNT: usize = 14;

#[derive(Clone, Debug, Eq, PartialEq, Serialize)]
struct DispatchRecord {
    ordinal: u32,
    phase: &'static str,
    stage: &'static str,
    kernel: &'static str,
}

#[derive(Clone, Debug, Eq, PartialEq)]
struct ImmutableFixedAbiEvidence {
    schema: String,
    status: String,
    document_sha256: String,
    manifest_document_sha256: String,
    admission_receipt_seal_sha256: String,
    static_execution_status: String,
    artifact_scan_or_payload_open_performed: bool,
    metal_context_or_dispatch_performed: bool,
    suffix_dispatches: Vec<DispatchRecord>,
    fixture_or_synthetic: bool,
}

#[derive(Clone, Debug, Eq, PartialEq)]
struct SealedOuterReceipt {
    schema: String,
    status: String,
    document_sha256: String,
    seal_sha256: String,
    receipt_written_last: bool,
    outer_reaped_child: bool,
}

#[derive(Clone, Debug, Eq, PartialEq)]
struct InnerReceipt {
    schema: String,
    status: String,
    document_sha256: String,
    strict_math: bool,
    device_execution_performed: bool,
    component_only: bool,
    complete_layer_or_token_performed: bool,
}

#[derive(Clone, Debug, Eq, PartialEq)]
struct SharedLineage {
    manifest_document_sha256: String,
    admission_receipt_seal_sha256: String,
    source_input_sha256: String,
    first_residual_buffer_sha256: String,
    token_command_buffer_sha256: String,
    route_plan_document_sha256: String,
}

#[derive(Clone, Debug, Eq, PartialEq)]
struct FirstResidualBridgeEvidence {
    outer: SealedOuterReceipt,
    inner: InnerReceipt,
    lineage: SharedLineage,
    source_input_kind: String,
    prefix_dispatches: Vec<DispatchRecord>,
    first_residual_device_parity_passed: bool,
    retained_device_buffer_elements: usize,
    fixture_or_synthetic: bool,
}

#[derive(Clone, Debug, Eq, PartialEq)]
struct RouteGuardReadback {
    value: u32,
    passed: bool,
    observed_ids: Vec<u32>,
    expected_ids: Vec<u32>,
}

#[derive(Clone, Debug, Eq, PartialEq)]
struct ParityWitness {
    label: String,
    elements: usize,
    source_input_sha256: String,
    first_residual_buffer_sha256: String,
    token_command_buffer_sha256: String,
    output_sha256: String,
    strict_math: bool,
    cpu_device_parity_passed: bool,
    fixture_or_synthetic: bool,
}

#[derive(Clone, Debug, Eq, PartialEq)]
struct TrueInputGraphEvidence {
    outer: SealedOuterReceipt,
    inner: InnerReceipt,
    lineage: SharedLineage,
    fixed_abi_document_sha256: String,
    first_residual_outer_document_sha256: String,
    first_residual_inner_document_sha256: String,
    ordered_dispatches: Vec<DispatchRecord>,
    route_guard: RouteGuardReadback,
    route_witnesses: Vec<ParityWitness>,
    shared_witness: ParityWitness,
    routed_sum_witness: ParityWitness,
    second_residual_witness: ParityWitness,
    command_buffer_fenced: bool,
    fixture_or_synthetic: bool,
}

#[derive(Clone, Debug, Eq, PartialEq)]
struct PostCaptureEvidence {
    fixed_abi: ImmutableFixedAbiEvidence,
    first_residual_bridge: FirstResidualBridgeEvidence,
    true_input_graph: TrueInputGraphEvidence,
}

#[derive(Clone, Debug, Eq, PartialEq)]
struct ComponentOnlyResult {
    status: &'static str,
    ordered_dispatch_count: usize,
    route_witness_count: usize,
}

fn prefix_dispatches() -> Vec<DispatchRecord> {
    [
        ("input_rmsnorm", "qwen_next_direct_packed_input_rmsnorm"),
        ("qkvz_projection", "qwen_binary_sign_scale_matvec"),
        ("ba_projection", "qwen_binary_sign_scale_matvec"),
        ("qkvz_rearrange_conv", "qwen_next_qkvz_rearrange_conv_l2"),
        ("ba_decay_beta", "qwen_next_ba_to_decay_beta"),
        ("deltanet_recurrent", "qwen_next_gated_delta_decode_single"),
        ("deltanet_gated_rmsnorm", "qwen_next_deltanet_gated_rmsnorm"),
        ("out_projection", "qwen_binary_sign_scale_matvec"),
        ("first_residual", "qwen_next_add_residual"),
    ]
    .into_iter()
    .enumerate()
    .map(|(index, (stage, kernel))| DispatchRecord {
        ordinal: (index + 1) as u32,
        phase: "deltanet_prefix",
        stage,
        kernel,
    })
    .collect()
}

fn fixed_suffix_dispatches() -> Vec<DispatchRecord> {
    [
        ("postnorm", "qwen80_postnorm_router_top10_rmsnorm"),
        ("router_matvec", "qwen80_postnorm_router_top10_matvec"),
        ("router_top10", "qwen80_postnorm_router_top10_select"),
        ("route_guard", "qwen80_all_ten_routed_wave_route_guard"),
        ("routed_gate_up", "qwen80_all_ten_routed_wave_gate_up"),
        ("routed_swiglu", "qwen80_all_ten_routed_wave_swiglu"),
        (
            "routed_down_weighted",
            "qwen80_all_ten_routed_wave_down_weighted",
        ),
        ("shared_gate_up", "qwen80_shared_expert_wave_gate_up"),
        ("shared_swiglu", "qwen80_shared_expert_wave_swiglu"),
        ("shared_down", "qwen80_shared_expert_wave_down"),
        (
            "shared_scalar_gate",
            "qwen80_shared_expert_wave_scalar_gate",
        ),
        (
            "shared_sigmoid",
            "qwen80_shared_expert_wave_apply_sigmoid_gate",
        ),
        (
            "routed_sum",
            "qwen80_moe_wave_aggregate_second_residual_route_sum",
        ),
        (
            "second_residual",
            "qwen80_moe_wave_aggregate_second_residual_add_shared_residual",
        ),
    ]
    .into_iter()
    .enumerate()
    .map(|(index, (stage, kernel))| DispatchRecord {
        ordinal: (index + 1) as u32,
        phase: "true_moe_suffix",
        stage,
        kernel,
    })
    .collect()
}

fn suffix_dispatches() -> Vec<DispatchRecord> {
    fixed_suffix_dispatches()
        .into_iter()
        .map(|mut dispatch| {
            dispatch.ordinal += PREFIX_DISPATCH_COUNT as u32;
            dispatch
        })
        .collect()
}

fn ordered_dispatches() -> Vec<DispatchRecord> {
    let mut full = prefix_dispatches();
    full.extend(suffix_dispatches());
    full
}

fn is_sha256(value: &str) -> bool {
    value.len() == 64
        && value.bytes().all(|byte| {
            byte.is_ascii_digit() || (byte.is_ascii_lowercase() && byte.is_ascii_hexdigit())
        })
}

fn require_sha256(value: &str, label: &str) -> Result<(), String> {
    if !is_sha256(value) {
        return Err(format!("{label} is not a lowercase SHA-256"));
    }
    Ok(())
}

fn validate_outer(
    receipt: &SealedOuterReceipt,
    schema: &str,
    status: &str,
    label: &str,
) -> Result<(), String> {
    if receipt.schema != schema || receipt.status != status {
        return Err(format!("{label} schema/status drifted"));
    }
    require_sha256(&receipt.document_sha256, &format!("{label} document"))?;
    require_sha256(&receipt.seal_sha256, &format!("{label} seal"))?;
    if !receipt.receipt_written_last || !receipt.outer_reaped_child {
        return Err(format!(
            "{label} lacks receipt-last outer-reaped durability"
        ));
    }
    Ok(())
}

fn validate_inner(
    receipt: &InnerReceipt,
    schema: &str,
    status: &str,
    label: &str,
) -> Result<(), String> {
    if receipt.schema != schema || receipt.status != status {
        return Err(format!("{label} schema/status drifted"));
    }
    require_sha256(&receipt.document_sha256, &format!("{label} document"))?;
    if !receipt.strict_math
        || !receipt.device_execution_performed
        || !receipt.component_only
        || receipt.complete_layer_or_token_performed
    {
        return Err(format!("{label} scope or strict-Math boundary drifted"));
    }
    Ok(())
}

fn validate_lineage(lineage: &SharedLineage, label: &str) -> Result<(), String> {
    require_sha256(
        &lineage.manifest_document_sha256,
        &format!("{label} manifest document"),
    )?;
    require_sha256(
        &lineage.admission_receipt_seal_sha256,
        &format!("{label} admission receipt seal"),
    )?;
    require_sha256(
        &lineage.source_input_sha256,
        &format!("{label} source input"),
    )?;
    require_sha256(
        &lineage.first_residual_buffer_sha256,
        &format!("{label} first residual"),
    )?;
    require_sha256(
        &lineage.token_command_buffer_sha256,
        &format!("{label} token command buffer"),
    )?;
    require_sha256(
        &lineage.route_plan_document_sha256,
        &format!("{label} route plan"),
    )?;
    Ok(())
}

fn validate_fixed_abi(fixed: &ImmutableFixedAbiEvidence) -> Result<(), String> {
    if fixed.schema != FIXED_ABI_SCHEMA || fixed.status != FIXED_ABI_STATUS {
        return Err("immutable fixed ABI schema/status drifted".into());
    }
    require_sha256(&fixed.document_sha256, "fixed ABI document")?;
    require_sha256(&fixed.manifest_document_sha256, "fixed ABI manifest")?;
    require_sha256(
        &fixed.admission_receipt_seal_sha256,
        "fixed ABI admission receipt seal",
    )?;
    if fixed.static_execution_status != "PREPARED_NOT_EXECUTED"
        || fixed.artifact_scan_or_payload_open_performed
        || fixed.metal_context_or_dispatch_performed
        || fixed.fixture_or_synthetic
    {
        return Err("fixed ABI must remain a real unexecuted non-fixture static plan".into());
    }
    if fixed.suffix_dispatches != fixed_suffix_dispatches() {
        return Err("fixed ABI does not retain the exact fourteen suffix dispatches".into());
    }
    Ok(())
}

fn validate_witness(
    witness: &ParityWitness,
    expected_label: &str,
    lineage: &SharedLineage,
) -> Result<(), String> {
    if witness.label != expected_label || witness.elements != HIDDEN {
        return Err(format!(
            "{expected_label} parity witness geometry/name drifted"
        ));
    }
    require_sha256(&witness.output_sha256, &format!("{expected_label} output"))?;
    if witness.source_input_sha256 != lineage.source_input_sha256
        || witness.first_residual_buffer_sha256 != lineage.first_residual_buffer_sha256
        || witness.token_command_buffer_sha256 != lineage.token_command_buffer_sha256
    {
        return Err(format!(
            "{expected_label} is not bound to the same input/hash/TCB"
        ));
    }
    if !witness.strict_math || !witness.cpu_device_parity_passed || witness.fixture_or_synthetic {
        return Err(format!(
            "{expected_label} parity is incomplete or fixture-derived"
        ));
    }
    Ok(())
}

/// The only success state is still component-only.  In particular this does
/// not authorize a complete layer, decoder, token, HCLI, TPS, or TG claim.
fn validate_post_capture(evidence: &PostCaptureEvidence) -> Result<ComponentOnlyResult, String> {
    validate_fixed_abi(&evidence.fixed_abi)?;

    let bridge = &evidence.first_residual_bridge;
    validate_outer(
        &bridge.outer,
        FIRST_RESIDUAL_OUTER_SCHEMA,
        FIRST_RESIDUAL_OUTER_STATUS,
        "first-residual outer",
    )?;
    validate_inner(
        &bridge.inner,
        FIRST_RESIDUAL_INNER_SCHEMA,
        FIRST_RESIDUAL_INNER_STATUS,
        "first-residual inner",
    )?;
    validate_lineage(&bridge.lineage, "first-residual bridge")?;
    if bridge.source_input_kind != "real_source_token"
        || bridge.fixture_or_synthetic
        || !bridge.first_residual_device_parity_passed
        || bridge.retained_device_buffer_elements != HIDDEN
    {
        return Err("first-residual bridge is not a real retained [2048] device witness".into());
    }
    if bridge.prefix_dispatches != prefix_dispatches() {
        return Err(
            "first-residual bridge does not retain the exact nine DeltaNet dispatches".into(),
        );
    }

    let graph = &evidence.true_input_graph;
    validate_outer(
        &graph.outer,
        TRUE_GRAPH_OUTER_SCHEMA,
        TRUE_GRAPH_OUTER_STATUS,
        "true-input graph outer",
    )?;
    validate_inner(
        &graph.inner,
        TRUE_GRAPH_INNER_SCHEMA,
        TRUE_GRAPH_INNER_STATUS,
        "true-input graph inner",
    )?;
    validate_lineage(&graph.lineage, "true-input graph")?;
    if bridge.lineage != graph.lineage {
        return Err("bridge and graph do not share the same input/hash/TCB lineage".into());
    }
    if evidence.fixed_abi.manifest_document_sha256 != bridge.lineage.manifest_document_sha256
        || evidence.fixed_abi.admission_receipt_seal_sha256
            != bridge.lineage.admission_receipt_seal_sha256
    {
        return Err("fixed ABI and first-residual bridge artifact authority drifted".into());
    }
    if graph.fixed_abi_document_sha256 != evidence.fixed_abi.document_sha256
        || graph.first_residual_outer_document_sha256 != bridge.outer.document_sha256
        || graph.first_residual_inner_document_sha256 != bridge.inner.document_sha256
    {
        return Err(
            "true-input graph does not bind immutable fixed ABI and bridge evidence".into(),
        );
    }
    if graph.fixture_or_synthetic || !graph.command_buffer_fenced {
        return Err("true-input graph is a fixture or lacks the common-TCB fence".into());
    }
    if graph.ordered_dispatches != ordered_dispatches() {
        return Err("true-input graph must retain the ordered 9+14 command graph".into());
    }
    if graph.route_guard.value != 1
        || !graph.route_guard.passed
        || graph.route_guard.observed_ids.len() != TOP_K
        || graph.route_guard.observed_ids != graph.route_guard.expected_ids
        || graph
            .route_guard
            .observed_ids
            .iter()
            .any(|&expert| expert >= 512)
    {
        return Err(
            "true-input graph route_guard!=1 or does not bind ten exact source routes".into(),
        );
    }
    let mut unique_route_ids = graph.route_guard.observed_ids.clone();
    unique_route_ids.sort_unstable();
    unique_route_ids.dedup();
    if unique_route_ids.len() != TOP_K {
        return Err("true-input graph top-10 route IDs are not distinct".into());
    }

    if graph.route_witnesses.len() != TOP_K {
        return Err("true-input graph lacks ten route parity witnesses".into());
    }
    for (index, witness) in graph.route_witnesses.iter().enumerate() {
        validate_witness(witness, &format!("route[{index}]"), &graph.lineage)?;
    }
    validate_witness(&graph.shared_witness, "gated_shared", &graph.lineage)?;
    validate_witness(&graph.routed_sum_witness, "routed_sum", &graph.lineage)?;
    validate_witness(
        &graph.second_residual_witness,
        "second_residual",
        &graph.lineage,
    )?;

    Ok(ComponentOnlyResult {
        status: COMPONENT_ONLY_RESULT,
        ordered_dispatch_count: PREFIX_DISPATCH_COUNT + SUFFIX_DISPATCH_COUNT,
        route_witness_count: TOP_K,
    })
}

fn prepared_document() -> serde_json::Value {
    json!({
        "schema": SCHEMA,
        "status": STATUS,
        "current_evidence_state": CURRENT_EVIDENCE_STATE,
        "mode": "cpu_only_post_capture_result_contract",
        "consumes_future_immutable_evidence": {
            "fixed_abi": {"schema": FIXED_ABI_SCHEMA, "status": FIXED_ABI_STATUS, "raw_document_sha256_required": true},
            "first_residual_bridge_outer": {"schema": FIRST_RESIDUAL_OUTER_SCHEMA, "status": FIRST_RESIDUAL_OUTER_STATUS, "sealed_outer_required": true},
            "first_residual_bridge_inner": {"schema": FIRST_RESIDUAL_INNER_SCHEMA, "status": FIRST_RESIDUAL_INNER_STATUS, "strict_math_device_component_required": true},
            "true_input_graph_outer": {"schema": TRUE_GRAPH_OUTER_SCHEMA, "status": TRUE_GRAPH_OUTER_STATUS, "sealed_outer_required": true},
            "true_input_graph_inner": {"schema": TRUE_GRAPH_INNER_SCHEMA, "status": TRUE_GRAPH_INNER_STATUS, "strict_math_device_component_required": true},
        },
        "same_lineage_required": [
            "manifest_document_sha256",
            "admission_receipt_seal_sha256",
            "source_input_sha256",
            "first_residual_buffer_sha256",
            "token_command_buffer_sha256",
            "route_plan_document_sha256",
        ],
        "required_command_graph": {
            "prefix_dispatches": PREFIX_DISPATCH_COUNT,
            "suffix_dispatches": SUFFIX_DISPATCH_COUNT,
            "total_dispatches": PREFIX_DISPATCH_COUNT + SUFFIX_DISPATCH_COUNT,
            "ordered_dispatches": ordered_dispatches(),
        },
        "required_readbacks": {
            "route_guard_value": 1,
            "route_guard_passed": true,
            "route_witnesses": TOP_K,
            "shared_expert_parity": true,
            "routed_sum_parity": true,
            "second_residual_parity": true,
            "all_witnesses_same_input_hash_and_tcb": true,
        },
        "refusals": [
            "materialized_source_route_shaped_fixture",
            "synthetic_or_fixture_input",
            "cpu_copy_substituted_for_retained_first_residual_device_buffer",
            "different_token_command_buffer_or_input_hash",
            "route_guard_not_equal_to_one",
            "missing_any_of_ten_route_shared_routed_sum_or_second_residual_parity_witnesses",
            "complete_layer_token_decoder_hcli_tps_tg_or_tournament_promotion",
        ],
        "claim_boundary": {
            "artifact_scan_or_payload_open_performed": false,
            "metal_context_or_dispatch_performed": false,
            "runtime_watcher_server_registry_or_hcli_changed": false,
            "current_result_is_not_a_capture": true,
            "not_a_complete_layer_token_decoder_generation_hcli_tps_tg_or_tournament_result": true,
        },
    })
}

fn main() {
    let arguments = env::args().skip(1).collect::<Vec<_>>();
    if !arguments.is_empty() && arguments != ["--print-plan"] {
        eprintln!(
            "usage: ascension_qwen80_l0_true_moe_post_capture_result_contract [--print-plan]"
        );
        std::process::exit(2);
    }
    println!(
        "{}",
        serde_json::to_string_pretty(&prepared_document())
            .expect("static Qwen80 L0 post-capture contract must serialize")
    );
}

#[cfg(test)]
mod tests {
    use super::*;

    fn hash(tag: u64) -> String {
        format!("{tag:064x}")
    }

    fn outer(schema: &str, status: &str, tag: u64) -> SealedOuterReceipt {
        SealedOuterReceipt {
            schema: schema.into(),
            status: status.into(),
            document_sha256: hash(tag),
            seal_sha256: hash(tag + 100),
            receipt_written_last: true,
            outer_reaped_child: true,
        }
    }

    fn inner(schema: &str, status: &str, tag: u64) -> InnerReceipt {
        InnerReceipt {
            schema: schema.into(),
            status: status.into(),
            document_sha256: hash(tag),
            strict_math: true,
            device_execution_performed: true,
            component_only: true,
            complete_layer_or_token_performed: false,
        }
    }

    fn lineage() -> SharedLineage {
        SharedLineage {
            manifest_document_sha256: hash(1),
            admission_receipt_seal_sha256: hash(2),
            source_input_sha256: hash(3),
            first_residual_buffer_sha256: hash(4),
            token_command_buffer_sha256: hash(5),
            route_plan_document_sha256: hash(6),
        }
    }

    fn witness(label: impl Into<String>, lineage: &SharedLineage, tag: u64) -> ParityWitness {
        ParityWitness {
            label: label.into(),
            elements: HIDDEN,
            source_input_sha256: lineage.source_input_sha256.clone(),
            first_residual_buffer_sha256: lineage.first_residual_buffer_sha256.clone(),
            token_command_buffer_sha256: lineage.token_command_buffer_sha256.clone(),
            output_sha256: hash(tag),
            strict_math: true,
            cpu_device_parity_passed: true,
            fixture_or_synthetic: false,
        }
    }

    /// This only models the shape of a future real capture so validator
    /// regressions can be tested. It is never emitted as present evidence.
    fn hypothetical_non_fixture_capture() -> PostCaptureEvidence {
        let lineage = lineage();
        let fixed = ImmutableFixedAbiEvidence {
            schema: FIXED_ABI_SCHEMA.into(),
            status: FIXED_ABI_STATUS.into(),
            document_sha256: hash(7),
            manifest_document_sha256: lineage.manifest_document_sha256.clone(),
            admission_receipt_seal_sha256: lineage.admission_receipt_seal_sha256.clone(),
            static_execution_status: "PREPARED_NOT_EXECUTED".into(),
            artifact_scan_or_payload_open_performed: false,
            metal_context_or_dispatch_performed: false,
            suffix_dispatches: fixed_suffix_dispatches(),
            fixture_or_synthetic: false,
        };
        let bridge = FirstResidualBridgeEvidence {
            outer: outer(FIRST_RESIDUAL_OUTER_SCHEMA, FIRST_RESIDUAL_OUTER_STATUS, 8),
            inner: inner(FIRST_RESIDUAL_INNER_SCHEMA, FIRST_RESIDUAL_INNER_STATUS, 9),
            lineage: lineage.clone(),
            source_input_kind: "real_source_token".into(),
            prefix_dispatches: prefix_dispatches(),
            first_residual_device_parity_passed: true,
            retained_device_buffer_elements: HIDDEN,
            fixture_or_synthetic: false,
        };
        let route_ids = vec![7, 30, 98, 87, 120, 45, 114, 53, 8, 91];
        let graph = TrueInputGraphEvidence {
            outer: outer(TRUE_GRAPH_OUTER_SCHEMA, TRUE_GRAPH_OUTER_STATUS, 10),
            inner: inner(TRUE_GRAPH_INNER_SCHEMA, TRUE_GRAPH_INNER_STATUS, 11),
            lineage: lineage.clone(),
            fixed_abi_document_sha256: fixed.document_sha256.clone(),
            first_residual_outer_document_sha256: bridge.outer.document_sha256.clone(),
            first_residual_inner_document_sha256: bridge.inner.document_sha256.clone(),
            ordered_dispatches: ordered_dispatches(),
            route_guard: RouteGuardReadback {
                value: 1,
                passed: true,
                observed_ids: route_ids.clone(),
                expected_ids: route_ids,
            },
            route_witnesses: (0..TOP_K)
                .map(|index| witness(format!("route[{index}]"), &lineage, 20 + index as u64))
                .collect(),
            shared_witness: witness("gated_shared", &lineage, 31),
            routed_sum_witness: witness("routed_sum", &lineage, 32),
            second_residual_witness: witness("second_residual", &lineage, 33),
            command_buffer_fenced: true,
            fixture_or_synthetic: false,
        };
        PostCaptureEvidence {
            fixed_abi: fixed,
            first_residual_bridge: bridge,
            true_input_graph: graph,
        }
    }

    #[test]
    fn current_contract_is_explicitly_prepared_and_incomplete() {
        let document = prepared_document();
        assert_eq!(document["schema"], SCHEMA);
        assert_eq!(document["status"], STATUS);
        assert_eq!(document["current_evidence_state"], CURRENT_EVIDENCE_STATE);
        assert_eq!(document["required_command_graph"]["prefix_dispatches"], 9);
        assert_eq!(document["required_command_graph"]["suffix_dispatches"], 14);
        assert_eq!(
            document["required_command_graph"]["ordered_dispatches"]
                .as_array()
                .unwrap()
                .len(),
            23
        );
        assert_eq!(
            document["claim_boundary"]["metal_context_or_dispatch_performed"],
            false
        );
    }

    #[test]
    fn hypothetical_non_fixture_capture_only_passes_as_component_result() {
        let result = validate_post_capture(&hypothetical_non_fixture_capture()).unwrap();
        assert_eq!(result.status, COMPONENT_ONLY_RESULT);
        assert_eq!(result.ordered_dispatch_count, 23);
        assert_eq!(result.route_witness_count, TOP_K);
    }

    #[test]
    fn rejects_different_token_command_buffer_even_when_other_hashes_match() {
        let mut evidence = hypothetical_non_fixture_capture();
        evidence
            .true_input_graph
            .lineage
            .token_command_buffer_sha256 = hash(99);
        let error = validate_post_capture(&evidence).unwrap_err();
        assert!(error.contains("same input/hash/TCB lineage"), "{error}");
    }

    #[test]
    fn rejects_route_guard_not_equal_to_one() {
        let mut evidence = hypothetical_non_fixture_capture();
        evidence.true_input_graph.route_guard.value = 0;
        let error = validate_post_capture(&evidence).unwrap_err();
        assert!(error.contains("route_guard!=1"));
    }

    #[test]
    fn rejects_ordered_command_graph_drift() {
        let mut evidence = hypothetical_non_fixture_capture();
        evidence.true_input_graph.ordered_dispatches.swap(8, 9);
        let error = validate_post_capture(&evidence).unwrap_err();
        assert!(error.contains("ordered 9+14"));
    }

    #[test]
    fn rejects_missing_all_ten_or_shared_or_second_residual_parity() {
        let mut evidence = hypothetical_non_fixture_capture();
        evidence.true_input_graph.route_witnesses.pop();
        let error = validate_post_capture(&evidence).unwrap_err();
        assert!(error.contains("ten route parity"));

        let mut evidence = hypothetical_non_fixture_capture();
        evidence
            .true_input_graph
            .shared_witness
            .cpu_device_parity_passed = false;
        let error = validate_post_capture(&evidence).unwrap_err();
        assert!(error.contains("gated_shared parity"));

        let mut evidence = hypothetical_non_fixture_capture();
        evidence
            .true_input_graph
            .second_residual_witness
            .cpu_device_parity_passed = false;
        let error = validate_post_capture(&evidence).unwrap_err();
        assert!(error.contains("second_residual parity"));
    }

    #[test]
    fn rejects_fixture_provenance_at_every_join_boundary() {
        let mut evidence = hypothetical_non_fixture_capture();
        evidence.fixed_abi.fixture_or_synthetic = true;
        let error = validate_post_capture(&evidence).unwrap_err();
        assert!(error.contains("non-fixture static plan"));

        let mut evidence = hypothetical_non_fixture_capture();
        evidence.first_residual_bridge.fixture_or_synthetic = true;
        let error = validate_post_capture(&evidence).unwrap_err();
        assert!(error.contains("real retained"));

        let mut evidence = hypothetical_non_fixture_capture();
        evidence.true_input_graph.route_witnesses[0].fixture_or_synthetic = true;
        let error = validate_post_capture(&evidence).unwrap_err();
        assert!(error.contains("route[0] parity"));
    }

    #[test]
    fn rejects_fixed_abi_or_bridge_identity_drift() {
        let mut evidence = hypothetical_non_fixture_capture();
        evidence.true_input_graph.fixed_abi_document_sha256 = hash(123);
        let error = validate_post_capture(&evidence).unwrap_err();
        assert!(error.contains("immutable fixed ABI"));

        let mut evidence = hypothetical_non_fixture_capture();
        evidence.first_residual_bridge.lineage.source_input_sha256 = hash(124);
        let error = validate_post_capture(&evidence).unwrap_err();
        assert!(error.contains("same input/hash/TCB lineage"), "{error}");
    }
}

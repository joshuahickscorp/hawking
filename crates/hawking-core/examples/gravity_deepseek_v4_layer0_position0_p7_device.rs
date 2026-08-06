//! Unsealed, bounded all-device DeepSeek-V4 layer-0 BOS P3A->P4A->P7->P6->P7 diagnostic.
//!
//! This is the prerequisite needed before a real layer-1 cache can exist: it
//! consumes the exact, device-resident BOS attention result from the existing
//! P3A->P4A continuation and computes the actual layer-0 FFN continuation on
//! the same Metal context.  It is deliberately not a decoder runtime, a
//! causal-cache proof, an endpoint, HCLI, generation, or TPS result.

#[cfg(not(target_os = "macos"))]
fn main() -> Result<(), Box<dyn std::error::Error>> {
    Err(std::io::Error::other(
        "gravity_deepseek_v4_layer0_position0_p7_device requires macOS Metal",
    )
    .into())
}

#[cfg(target_os = "macos")]
mod macos {
    use hawking_core::gravity_deepseek_v4::DeepSeekV4FullStreamReader;
    use hawking_core::gravity_deepseek_v4_execution_context::{
        DeepSeekV4ExecutionContext, DeepSeekV4ExecutionContextConfig, DeepSeekV4SelectedRouteSet,
    };
    use hawking_core::gravity_deepseek_v4_expert_cache::{
        resolve_expert_bundle, DeepSeekV4ExpertBundleCache, ExpertBundleKey,
    };
    use hawking_core::gravity_deepseek_v4_layer0_attention::hc_attn_post_source_algorithm;
    use hawking_core::gravity_deepseek_v4_layer0_moe::{
        layer0_mhc_ffn_control_f64_authority, layer0_mhc_ffn_post_f64_authority,
        layer0_moe_body_f32_oracle_for_token,
        layer0_moe_body_f32_oracle_from_verified_gate_logits_for_token,
        layer0_moe_body_f64_authority_for_token, layer0_moe_successor_cpu_oracle,
        Layer0MoeCombineOrder,
    };
    use hawking_core::gravity_deepseek_v4_layer0_prefix::{
        HC_FLAT_WIDTH, HC_MIX_WIDTH, HC_MULT, HIDDEN_SIZE, PREFIX_TOKEN_ID,
    };
    use hawking_core::gravity_deepseek_v4_layer_scheduler::{
        DeepSeekV4LayerPreparationResult, DeepSeekV4LayerPreparationScheduler,
        DeepSeekV4LayerPreparationStage,
    };
    use hawking_core::gravity_deepseek_v4_p0_gate_calibration::{
        layer0_moe_body_f32_oracle_from_qualified_torch_route_calibration_for_token,
        load_verified_p0_gate_torch_f32_calibration,
        load_verified_p0_gate_torch_f32_route_calibration, DeepSeekV4P0GateTorchF32Calibration,
        DeepSeekV4P0GateTorchF32RouteCalibration, P0_GATE_TORCH_F32_ROUTE_CALIBRATION_SCHEMA,
    };
    use hawking_core::gravity_deepseek_v4_p3a_stage_sink::{
        DeepSeekV4P3aMetalStageSink, DeepSeekV4P4aContinuationPhase,
        DeepSeekV4P4aContinuationReport, DSV4F_P3A_Q_CHAIN_DISPATCHES,
    };
    use hawking_core::gravity_deepseek_v4_p6_device::{
        DeepSeekV4Layer0P6MetalExecutor, DSV4F_P6_DEVICE_COMMAND_BUFFERS,
        DSV4F_P6_DEVICE_COMPUTE_ENCODERS, DSV4F_P6_DEVICE_CPU_VISIBLE_WAITS,
        DSV4F_P6_DEVICE_DISPATCHES, P6_C4_GATE_GRID_THREADS, P6_C4_GATE_KERNEL,
        P6_C4_GATE_SIMDGROUP_THREADS,
    };
    use hawking_core::gravity_deepseek_v4_p7_composition::{
        DeepSeekV4P7AttentionDeviceState, DeepSeekV4P7SourceLeasePreparation,
        DeepSeekV4P7SourceTensorBinding,
    };
    use hawking_core::gravity_deepseek_v4_p7_device::{
        DeepSeekV4P7BoundedDeviceExecutor, DSV4F_P7_OWNED_COMMAND_BUFFERS,
        DSV4F_P7_OWNED_COMPUTE_ENCODERS, DSV4F_P7_OWNED_CPU_VISIBLE_WAITS,
        DSV4F_P7_OWNED_DEVICE_DISPATCHES,
    };
    use hawking_core::metal::{PhysicalTraceGuard, PhysicalTraceIdentity};
    use hawking_core::numeric_parity::{score_pair, Bounds, PairedScore};
    use serde_json::{json, Value};
    use sha2::{Digest, Sha256};
    use std::error::Error;
    use std::fs::{self, File, OpenOptions};
    use std::io::Write;
    use std::path::{Path, PathBuf};

    const DIAGNOSTIC_SCHEMA: &str =
        "hawking.gravity.deepseek_v4.p7_layer0_position0_device_diagnostic.v1";
    const GATE_INPUT_TRACE_SCHEMA: &str =
        "hawking.gravity.deepseek_v4.p7_layer0_position0_gate_input_trace.v1";
    const GATE_INPUT_TRACE_STATUS: &str =
        "UNSEALED_POST_COMPLETION_BOS_FFN_NORM_GATE_INPUT_TRACE_NON_RECEIPT";
    const GATE_INPUT_TRACE_RAW_PAYLOAD_MAX_BYTES: usize = HIDDEN_SIZE * std::mem::size_of::<u16>();
    const GATE_INPUT_TRACE_SERIALIZED_MAX_BYTES: usize = 64 * 1024;
    const STATUS: &str = "P7_LAYER0_BOS_P3A_P4A_EXACT_ATTENTION_REAL_METAL_DEVICE_GRAPH_UNSEALED_DIAGNOSTIC_NUMERIC_PARITY_V2_1_ONLY_NOT_RUNTIME";
    const STRICT_MATH_CANDIDATE_STATUS: &str = "P7_LAYER0_BOS_P3A_P4A_STRICT_MATH_CANDIDATE_REAL_METAL_DEVICE_GRAPH_UNSEALED_DIAGNOSTIC_NUMERIC_PARITY_V2_1_ONLY_NOT_RUNTIME";
    const EXPECTED_ROUTE_IDS_BY_TOP_SLOT: [u16; 6] = [254, 222, 245, 200, 53, 35];
    const EXPECTED_NUMERIC_COMBINE_ORDER: [(u32, u32); 6] =
        [(5, 35), (4, 53), (3, 200), (1, 222), (2, 245), (0, 254)];

    type ProbeResult<T> = Result<T, Box<dyn Error>>;

    struct Args {
        artifact: PathBuf,
        out: PathBuf,
        gate_input_trace_out: Option<PathBuf>,
        source_gate_calibration: Option<PathBuf>,
        source_gate_route_calibration: Option<PathBuf>,
        strict_math: bool,
    }

    pub fn run() -> ProbeResult<()> {
        let args = parse_args()?;
        let (selected_status, compile_math_mode, trace_phase, gate_candidate_status) = if args
            .strict_math
        {
            (
                    STRICT_MATH_CANDIDATE_STATUS,
                    "strict_math_candidate_fast_math_disabled_for_full_linked_library",
                    "dsv4f_p7_layer0_position0_strict_math_candidate",
                    "strict-math candidate selected: P3A/P4A/P7/P6 use one opt-in Metal library compiled with fast math disabled; this is a comparison surface only and does not isolate or promote the Gate kernel alone",
                )
        } else {
            (
                    STATUS,
                    "baseline_default_metal_compile_options",
                    "dsv4f_p7_layer0_position0",
                    "baseline selected: no strict source-order Gate candidate was selected or promoted by this failure report",
                )
        };
        let mut context = DeepSeekV4ExecutionContext::open(
            &args.artifact,
            DeepSeekV4ExecutionContextConfig::default(),
        )?;
        // An optional qualified target is loaded and bound to the admitted
        // artifact before any P3A/P4A/P7/P6 Metal graph is constructed. It is
        // still inert until the post-completion BF16 input hash check below.
        let source_gate_calibration = args
            .source_gate_calibration
            .as_ref()
            .map(|path| load_verified_p0_gate_torch_f32_calibration(context.spine().reader(), path))
            .transpose()?;
        let source_gate_route_calibration = args
            .source_gate_route_calibration
            .as_ref()
            .map(|path| {
                load_verified_p0_gate_torch_f32_route_calibration(context.spine().reader(), path)
            })
            .transpose()?;
        let prepared = context.prepare_decode_input(PREFIX_TOKEN_ID as u32)?;
        if prepared.token_id != PREFIX_TOKEN_ID as u32 || prepared.position != 0 {
            return Err("BOS preparation did not bind layer-0 token 0 / position 0".into());
        }
        let route_set = DeepSeekV4SelectedRouteSet::new(EXPECTED_ROUTE_IDS_BY_TOP_SLOT)?;

        // The existing P3A/P4A continuation owns its Metal context and keeps
        // the completed BF16[4,4096] attention result resident in that context.
        let mut attention_sink = if args.strict_math {
            DeepSeekV4P3aMetalStageSink::new_for_verified_p4a_continuation_strict_math(
                &context, &prepared,
            )?
        } else {
            DeepSeekV4P3aMetalStageSink::new_for_verified_p4a_continuation(&context, &prepared)?
        };
        let mut attention_scheduler =
            DeepSeekV4LayerPreparationScheduler::new(&context, 0, route_set)?;
        let expected_attention_stages = [
            DeepSeekV4LayerPreparationStage::MhcAttentionControl,
            DeepSeekV4LayerPreparationStage::AttentionControl(
                hawking_core::gravity_deepseek_v4_runtime_spine::DeepSeekV4ControlProjection::WqA,
            ),
            DeepSeekV4LayerPreparationStage::AttentionControl(
                hawking_core::gravity_deepseek_v4_runtime_spine::DeepSeekV4ControlProjection::WqB,
            ),
            DeepSeekV4LayerPreparationStage::AttentionControl(
                hawking_core::gravity_deepseek_v4_runtime_spine::DeepSeekV4ControlProjection::Wkv,
            ),
            DeepSeekV4LayerPreparationStage::AttentionControl(
                hawking_core::gravity_deepseek_v4_runtime_spine::DeepSeekV4ControlProjection::WoA,
            ),
            DeepSeekV4LayerPreparationStage::AttentionControl(
                hawking_core::gravity_deepseek_v4_runtime_spine::DeepSeekV4ControlProjection::WoB,
            ),
        ];
        for expected in expected_attention_stages {
            let step = attention_scheduler
                .execute_next_with_sink(&mut context, &mut attention_sink)?
                .ok_or("P3A/P4A scheduler ended before the BOS attention continuation completed")?;
            if step.stage != expected {
                return Err("P3A/P4A scheduler produced an unexpected attention stage".into());
            }
        }
        let p4a_report = attention_sink.finish_p4a_continuation()?;
        validate_p4a_report(&p4a_report)?;

        // The FFN static controls still come through the live source-stage
        // scheduler. This is staging only; all activation data remains in the
        // P3A/P4A-owned Metal context.
        let source_preparation = stage_p7_source_lease(&mut context, &prepared, route_set)?;
        let source = source_preparation.source_contract()?;
        if source.layer != 0
            || source.token_id != PREFIX_TOKEN_ID as u32
            || source.token_position != 0
            || source.host_activation_handoff_permitted
        {
            return Err("P7 source lease did not retain the BOS no-host boundary".into());
        }
        let full_causal_execution_denied = context.require_full_causal_execution().is_err();
        if !full_causal_execution_denied {
            return Err(
                "bounded P0 diagnostic unexpectedly admitted a full causal executor".into(),
            );
        }

        let reader = context.spine().reader();
        let required_hot_bytes = required_hot_cache_bytes(reader, &EXPECTED_ROUTE_IDS_BY_TOP_SLOT)?;
        let mut cache = DeepSeekV4ExpertBundleCache::new(required_hot_bytes, 0)?;
        let metal = attention_sink.metal_context();
        let p6 =
            DeepSeekV4Layer0P6MetalExecutor::prepare_for_p7(metal, reader, &mut cache, &source)?;
        let p6_bindings = p6.source_bindings().clone();
        let expected_ids_u32 = EXPECTED_ROUTE_IDS_BY_TOP_SLOT.map(u32::from);
        if p6_bindings.selected_expert_ids_top_slot_order != expected_ids_u32 {
            return Err("P6 static BOS tid2eid plan differs from the admitted source row".into());
        }
        let numeric_order: Vec<(u32, u32)> = p6_bindings
            .resident_experts_numeric_source_order
            .iter()
            .map(|binding| (binding.source_top_slot, binding.expert_id))
            .collect();
        if numeric_order.as_slice() != EXPECTED_NUMERIC_COMBINE_ORDER {
            return Err(
                "P6 BOS resident-expert order does not preserve numeric source order".into(),
            );
        }
        let cache_after_prepare = cache.state();
        let mut p7 = DeepSeekV4P7BoundedDeviceExecutor::prepare_from_source_lease(
            metal,
            &source_preparation,
            Box::new(p6),
        )?;

        // Do not mix preparation accounting into the real P7/P6 graph. P3A
        // and P4A have already frozen their own completed accounting above.
        let _ = metal.drain_trace();
        let _ = metal.drain_stats();
        let trace_identity = PhysicalTraceIdentity::new(
            sha256(b"dsv4f-p7-layer0-bos-p0-unsealed-diagnostic"),
            sha256(reader.manifest_seal_sha256().as_bytes()),
            trace_phase.to_owned(),
            "p3a_p4a_to_p7_p6_p7_ffn_continuation".to_owned(),
            Some(1),
            0,
        )?;
        let physical_trace = PhysicalTraceGuard::begin(trace_identity)?;
        let attention = DeepSeekV4P7AttentionDeviceState::position0(
            metal,
            attention_sink.p4a_attention_output_buffer()?,
            0,
            PREFIX_TOKEN_ID as u32,
        )?;
        let output = p7.execute_position0(attention)?;
        output.validate()?;
        // The strict library changes every linked shader's compile profile.
        // It is useful for a Gate arithmetic comparison, but it cannot borrow
        // the baseline P4A exact-attention label until P4A has been separately
        // revalidated under that profile.
        let (
            p4a_predecessor_exact,
            p4a_predecessor_label,
            p4a_predecessor_relation,
            mhc_pre_norm_reference,
            mhc_post_child_reference,
            claim_boundary,
        ) = if args.strict_math {
            (
                false,
                "P4A_UNREVALIDATED_STRICT_MATH_CANDIDATE",
                "P3A-to-P4A BOS attention device buffer borrowed directly by P7 under the strict-math candidate; no host activation or fabricated KV cache was introduced, but this altered compile profile has not revalidated P4A exact-attention parity",
                "independent FP64 mHC linear/Sinkhorn accumulation from the captured P3A-to-P4A BF16 state under the un-revalidated strict-math candidate and verified F32 controls",
                "same captured P3A-to-P4A strict-math-candidate BF16 state plus same device-produced P6 MoE BF16 row; independent source-F32 and FP64 mHC-post",
                "A real same-context Metal P0 FFN strict-math candidate executed from a P3A-to-P4A BOS attention buffer: mHC-FFN pre/norm, hash Gate, six native-FP4 routed experts, one native-FP8 shared expert, source-order combine, and mHC-FFN post. The altered strict-math P4A predecessor is explicitly un-revalidated and carries no exact-attention claim; P7 is reported only through the explicit same-input Numeric Parity V2.1 diagnostics above. This unsealed diagnostic is not a causal KV continuation, layer-1 execution, registered 43-layer runtime, HCLI endpoint, generation, first-token, or TPS claim.",
            )
        } else {
            (
                output.p4b_predecessor_parity.predecessor_attention_is_exact(),
                output.p4b_predecessor_parity.as_str(),
                "exact P3A-to-P4A BOS attention device buffer borrowed directly by P7; no host activation or fabricated KV cache was introduced",
                "independent FP64 mHC linear/Sinkhorn accumulation from captured exact P4A BF16 state and verified F32 controls",
                "same exact P4A attention BF16 state plus same device-produced P6 MoE BF16 row; independent source-F32 and FP64 mHC-post",
                "A real same-context Metal P0 FFN continuation executed from the exact P3A-to-P4A BOS attention buffer: mHC-FFN pre/norm, hash Gate, six native-FP4 routed experts, one native-FP8 shared expert, source-order combine, and mHC-FFN post. The P4A attention predecessor remains exact-only at its own boundary; P7 is reported only through the explicit same-input Numeric Parity V2.1 diagnostics above. This unsealed diagnostic is not a causal KV continuation, layer-1 execution, registered 43-layer runtime, HCLI endpoint, generation, first-token, or TPS claim.",
            )
        };
        let physical_counts = physical_trace.counts();
        drop(physical_trace);

        // All graph command buffers have completed. Every following read is a
        // post-completion diagnostic, never a bridge into P7 or P6.
        let attention_hc_post =
            read_gpu_u16(attention_sink.p4a_attention_output_buffer()?, HC_FLAT_WIDTH)?;
        let ffn_reduced = read_gpu_u16(&output.ffn_reduced_bf16, HIDDEN_SIZE)?;
        let ffn_norm = read_gpu_u16(&output.ffn_norm_bf16, HIDDEN_SIZE)?;
        let mhc_flat_rsqrt = read_gpu_f32(&output.mhc_flat_rsqrt_f32, 1)?;
        let mhc_mixes = read_gpu_f32(&output.mhc_mixes_f32, HC_MIX_WIDTH)?;
        let mhc_pre = read_gpu_f32(&output.mhc_pre_f32, HC_MULT)?;
        let mhc_post = read_gpu_f32(&output.mhc_post_f32, HC_MULT)?;
        let mhc_comb = read_gpu_f32(&output.mhc_comb_f32, HC_MULT * HC_MULT)?;
        let moe = read_gpu_u16(&output.p6.moe_output_bf16, HIDDEN_SIZE)?;
        let child = read_gpu_u16(&output.child_hc_state_bf16, HC_FLAT_WIDTH)?;
        let route_ids = read_gpu_u32(&output.p6.route_ids_u32, 6)?;
        let route_weights = read_gpu_f32(&output.p6.route_weights_f32, 6)?;
        let gate_logits = read_gpu_f32(&output.p6.gate_logits_f32, 256)?;
        let original_scores = read_gpu_f32(&output.p6.original_scores_f32, 256)?;
        let route_valid = read_gpu_u32(&output.p6.route_valid_u32, 1)?;

        // This must remain after the real graph has completed and before a
        // qualified Torch target may influence the CPU diagnostic route/MoE.
        // A mismatch fails closed; no calibrated comparison or output follows.
        match (
            source_gate_route_calibration.as_ref(),
            source_gate_calibration.as_ref(),
        ) {
            (Some(calibration), None) => {
                calibration.validate_observed_gate_input_bf16(&ffn_norm)?
            }
            (None, Some(calibration)) => {
                calibration.validate_observed_gate_input_bf16(&ffn_norm)?
            }
            (None, None) => {}
            (Some(_), Some(_)) => {
                return Err("v1 and v2 Gate calibrations must remain mutually exclusive".into())
            }
        }
        let source_gate_reference = source_gate_reference_json(
            source_gate_calibration.as_ref(),
            source_gate_route_calibration.as_ref(),
        )?;

        // This optional trace is deliberately emitted only after the real
        // graph has completed and every diagnostic readback above is already
        // resident on the host. It is not consumed by P7/P6, does not touch
        // Metal accounting, and remains available when the later V2.1 gate
        // rejects this run. The trace contains one bounded raw activation so
        // a Gate accumulation experiment can be reproduced without retaining
        // any source weights or additional hidden-state payloads.
        let gate_input_trace = match args.gate_input_trace_out.as_ref() {
            Some(path) => Some(write_new_gate_input_trace(
                path,
                reader,
                &source,
                &p6_bindings,
                &attention_hc_post,
                &ffn_norm,
                &gate_logits,
                &metal.device_name(),
                compile_math_mode,
                args.strict_math,
                p4a_predecessor_exact,
                p4a_predecessor_label,
            )?),
            None => None,
        };

        let same_input_successor =
            layer0_moe_successor_cpu_oracle(reader, PREFIX_TOKEN_ID, &attention_hc_post)?;
        let same_input_mhc = &same_input_successor.ffn_hc_pre;
        let same_input_mhc_f64 = layer0_mhc_ffn_control_f64_authority(reader, &attention_hc_post)?;
        let control_bounds = p7_mhc_control_v21_bounds();
        let storage_bounds = p7_bf16_storage_v21_bounds();
        let mhc_flat_rsqrt_v21 = score_pair(
            &[same_input_mhc.flat_rsqrt],
            &mhc_flat_rsqrt,
            &[same_input_mhc_f64.flat_rsqrt_f64],
            &control_bounds,
        );
        let mhc_mixes_v21 = score_pair(
            &same_input_mhc.mixes_f32,
            &mhc_mixes,
            &same_input_mhc_f64.mixes_f64,
            &control_bounds,
        );
        let mhc_pre_v21 = score_pair(
            &same_input_mhc.pre_f32,
            &mhc_pre,
            &same_input_mhc_f64.pre_f64,
            &control_bounds,
        );
        let mhc_post_v21 = score_pair(
            &same_input_mhc.post_f32,
            &mhc_post,
            &same_input_mhc_f64.post_f64,
            &control_bounds,
        );
        let mhc_comb_v21 = score_pair(
            &same_input_mhc.comb_f32,
            &mhc_comb,
            &same_input_mhc_f64.comb_f64,
            &control_bounds,
        );
        let mhc_reduced_v21 = score_pair(
            &bf16_bits_f32(&same_input_mhc.reduced_bf16_bits),
            &bf16_bits_f32(&ffn_reduced),
            &bf16_bits_f64(&same_input_mhc_f64.reduced_bf16_bits),
            &storage_bounds,
        );
        let ffn_norm_source_f32_exact = same_input_successor.ffn_norm_bf16_bits == ffn_norm;
        let ffn_norm_source_f32_store =
            bf16_store_mismatch_summary(&same_input_successor.ffn_norm_bf16_bits, &ffn_norm);
        let mhc_v21_pass = mhc_flat_rsqrt_v21.pass
            && mhc_mixes_v21.pass
            && mhc_pre_v21.pass
            && mhc_post_v21.pass
            && mhc_comb_v21.pass
            && mhc_reduced_v21.pass
            && ffn_norm_source_f32_exact;

        let same_input_moe_cpu = match (
            source_gate_route_calibration.as_ref(),
            source_gate_calibration.as_ref(),
        ) {
            (Some(calibration), None) => {
                layer0_moe_body_f32_oracle_from_qualified_torch_route_calibration_for_token(
                    reader,
                    PREFIX_TOKEN_ID,
                    &ffn_norm,
                    calibration,
                )?
            }
            (None, Some(calibration)) => {
                layer0_moe_body_f32_oracle_from_verified_gate_logits_for_token(
                    reader,
                    PREFIX_TOKEN_ID,
                    &ffn_norm,
                    calibration.logits_f32(),
                )?
            }
            (None, None) => {
                layer0_moe_body_f32_oracle_for_token(reader, PREFIX_TOKEN_ID, &ffn_norm)?
            }
            (Some(_), Some(_)) => {
                return Err("v1 and v2 Gate calibrations must remain mutually exclusive".into())
            }
        };
        let same_input_moe_f64 =
            layer0_moe_body_f64_authority_for_token(reader, PREFIX_TOKEN_ID, &ffn_norm)?;
        let same_input_route = &same_input_moe_cpu.route;
        let same_input_f64_route = &same_input_moe_f64.route;
        let same_input_ids = ids_u32(&same_input_route.selected_expert_ids)?;
        let f64_ids = ids_u32(&same_input_f64_route.selected_expert_ids)?;
        let source_f32_order = source_combine_order_u32(&same_input_moe_cpu.routed_combine_order)?;
        let f64_order = source_combine_order_u32(&same_input_moe_f64.routed_combine_order)?;
        if same_input_ids != expected_ids_u32
            || f64_ids != expected_ids_u32
            || source_f32_order.as_slice() != EXPECTED_NUMERIC_COMBINE_ORDER
            || f64_order.as_slice() != EXPECTED_NUMERIC_COMBINE_ORDER
        {
            return Err(
                "same-input CPU/F64 BOS route disagrees with the verified static route plan".into(),
            );
        }
        let route_bounds = Bounds {
            max_meaningful_rel: 1.0e-4,
            ..Bounds::continuous_only()
        };
        let gate_logits_v21 = score_pair(
            &same_input_route.logits_f32,
            &gate_logits,
            &same_input_f64_route.logits_f64,
            &route_bounds,
        );
        let original_scores_v21 = score_pair(
            &same_input_route.original_scores_f32,
            &original_scores,
            &same_input_f64_route.original_scores_f64,
            &route_bounds,
        );
        let route_weights_v21 = score_pair(
            &same_input_route.selected_weights_f32,
            &route_weights,
            &same_input_f64_route.selected_weights_f64,
            &route_bounds,
        );
        let route_v21_pass = gate_logits_v21.pass
            && original_scores_v21.pass
            && route_weights_v21.pass
            && same_input_ids == route_ids
            && route_valid == vec![1];
        let direct_upstream_gate_route_parity = direct_upstream_gate_route_parity_json(
            source_gate_route_calibration.as_ref(),
            &gate_logits_v21,
            &original_scores_v21,
            &route_weights_v21,
            same_input_ids == route_ids,
            route_valid == vec![1],
        );
        let moe_v21 = score_pair(
            &bf16_bits_f32(&same_input_moe_cpu.moe_output_bf16_bits),
            &bf16_bits_f32(&moe),
            &same_input_moe_f64.combined_f64,
            &storage_bounds,
        );
        let child_cpu = hc_attn_post_source_algorithm(
            &moe,
            &attention_hc_post,
            &same_input_mhc.post_f32,
            &same_input_mhc.comb_f32,
        )?;
        let child_f64 = layer0_mhc_ffn_post_f64_authority(reader, &attention_hc_post, &moe)?;
        let child_v21 = score_pair(
            &bf16_bits_f32(&child_cpu),
            &bf16_bits_f32(&child),
            &child_f64.child_state_f64,
            &storage_bounds,
        );
        let all_same_input_v21_pass =
            mhc_v21_pass && route_v21_pass && moe_v21.pass && child_v21.pass;
        if !all_same_input_v21_pass {
            let gate_logit_top_mismatches = bounded_gate_logit_mismatch_diagnostics(
                &same_input_route.logits_f32,
                &gate_logits,
                &same_input_f64_route.logits_f64,
                gate_logits_v21.abs_error_cutoff,
            )?;
            // This path deliberately reports only bounded score summaries to
            // stderr and returns before the create-new writer below. It is a
            // debugging observation, not an unsealed diagnostic artifact or
            // a receipt that could be promoted.
            let failure = json!({
                "schema": "hawking.gravity.deepseek_v4.p7_layer0_position0_failure_only.v1",
                "status": "FAIL_SAME_INPUT_P0_CPU_F64_NUMERIC_PARITY_V2_1_NO_RECEIPT_OR_OUTPUT_EMITTED",
                "unsealed": true,
                "failure_only": true,
                "receipt_promoted": false,
                "output_path_written": false,
                "normal_output_path_written": false,
                "requested_output_path": args.out.display().to_string(),
                "optional_gate_input_trace": &gate_input_trace,
                "scope": {
                    "layer": output.layer,
                    "token_id": output.token_id,
                    "token_position": output.token_position,
                    "p4a_attention_predecessor_exact": p4a_predecessor_exact,
                    "p4a_attention_predecessor_label": p4a_predecessor_label,
                    "p7_exact_storage_claim": false,
                    "runtime_claim": false,
                    "compile_math_mode": compile_math_mode,
                },
                "source_gate_reference": &source_gate_reference,
                "post_completion_only": true,
                "boundaries": {
                    "mhc_controls_and_store": {
                        "control_bounds": control_bounds,
                        "storage_bounds": storage_bounds,
                        "flat_rsqrt": mhc_flat_rsqrt_v21,
                        "mixes": mhc_mixes_v21,
                        "pre": mhc_pre_v21,
                        "post": mhc_post_v21,
                        "comb": mhc_comb_v21,
                        "reduced_bf16_store": mhc_reduced_v21,
                        "ffn_norm_source_f32_store": ffn_norm_source_f32_store,
                        "pass": mhc_v21_pass,
                    },
                    "route_ids_and_weights": {
                        "bounds": route_bounds,
                        "gate_logits": gate_logits_v21,
                        "gate_logit_top_mismatches": gate_logit_top_mismatches,
                        "gate_logit_accumulation_semantics": {
                            "source_f32": source_gate_cpu_semantics(
                                source_gate_calibration.as_ref(),
                                source_gate_route_calibration.as_ref(),
                            ),
                            "baseline_device": "deepseek_v4_p0_gate_reduction_c4_simd32_fma_candidate: one 32-thread SIMDgroup per Gate row; lane i accumulates i, i+32, ... with precise FMA, then Metal simd_sum reduces the row. It was admitted only by the isolated frozen P0 Gate sweep; this bounded P7/P6 graph still requires its own route/MoE parity proof and makes no runtime or TPS claim.",
                            "candidate_status": gate_candidate_status,
                        },
                        "original_scores": original_scores_v21,
                        "selected_weights": route_weights_v21,
                        "expected_top_slot_ids": expected_ids_u32,
                        "source_f32_top_slot_ids": same_input_ids,
                        "fp64_top_slot_ids": f64_ids,
                        "device_top_slot_ids": route_ids,
                        "route_valid_word": route_valid,
                        "source_f32_numeric_combine_order": source_f32_order,
                        "fp64_numeric_combine_order": f64_order,
                        "pass": route_v21_pass,
                    },
                    "direct_upstream_gate_route_parity": &direct_upstream_gate_route_parity,
                    "moe_bf16_store": {
                        "storage_bounds": storage_bounds,
                        "score": moe_v21,
                        "pass": moe_v21.pass,
                    },
                    "mhc_post_child_bf16_store": {
                        "storage_bounds": storage_bounds,
                        "score": child_v21,
                        "pass": child_v21.pass,
                    },
                },
                "observed_hashes": {
                    "p4a_attention_hc_post_bf16": sha256_u16(&attention_hc_post),
                    "p7_ffn_reduced_bf16": sha256_u16(&ffn_reduced),
                    "p7_ffn_norm_bf16": sha256_u16(&ffn_norm),
                    "p6_moe_output_bf16": sha256_u16(&moe),
                    "p7_child_hc_state_bf16": sha256_u16(&child),
                },
                "all_scored_sections_pass": false,
                "next_action": "inspect the failing bounded score(s); do not relax thresholds, change route constants, write an artifact, or promote a receipt from this report",
            });
            eprintln!(
                "{}",
                serde_json::to_string_pretty(&decimal_strings(failure))?
            );
            return Err("same-input P0 CPU/F64 Numeric Parity V2.1 gate did not pass".into());
        }

        let trace = metal.drain_trace();
        let (graph_buffers_created, graph_bytes_allocated, committed_command_buffers) =
            metal.drain_stats();
        let expected_command_buffers =
            DSV4F_P7_OWNED_COMMAND_BUFFERS + DSV4F_P6_DEVICE_COMMAND_BUFFERS;
        let expected_waits = DSV4F_P7_OWNED_CPU_VISIBLE_WAITS + DSV4F_P6_DEVICE_CPU_VISIBLE_WAITS;
        let expected_dispatches = DSV4F_P7_OWNED_DEVICE_DISPATCHES + DSV4F_P6_DEVICE_DISPATCHES;
        let expected_encoders = DSV4F_P7_OWNED_COMPUTE_ENCODERS + DSV4F_P6_DEVICE_COMPUTE_ENCODERS;
        if committed_command_buffers != expected_command_buffers
            || trace.len() != expected_command_buffers
            || physical_counts.command_count as usize != expected_command_buffers
            || physical_counts.encoder_count as usize != expected_encoders
            || route_valid != vec![1]
        {
            return Err("P0 P7/P6 command topology or route-validity contract failed".into());
        }

        let unsigned = json!({
            "schema": DIAGNOSTIC_SCHEMA,
            "status": selected_status,
            "unsealed": true,
            "optional_gate_input_trace": &gate_input_trace,
            "artifact": {
                "path": reader.artifact_root().display().to_string(),
                "manifest_seal_sha256": reader.manifest_seal_sha256(),
                "manifest_file_sha256": reader.manifest_file_sha256(),
                "restart_receipt_seal_sha256": reader.restart_seal_sha256(),
                "source_repository": reader.source_identity().repository,
                "source_revision": reader.source_identity().revision,
                "source_parent_retained": false,
            },
            "scope": {
                "layer": output.layer,
                "token_id": output.token_id,
                "token_position": output.token_position,
                "p4a_attention_predecessor_exact": p4a_predecessor_exact,
                "p4a_attention_predecessor_label": p4a_predecessor_label,
                "p7_exact_storage_claim": false,
                "registered_43_layer_engine": false,
                "causal_forward": false,
                "causal_kv_persistence": false,
                "hcli_endpoint": false,
                "base_true_tps_eligible": false,
                "full_causal_execution_denied": full_causal_execution_denied,
                "compile_math_mode": compile_math_mode,
                "strict_math_candidate": args.strict_math,
            },
            "p3a_p4a_predecessor": {
                "phase": p4a_report.phase.as_str(),
                "p3a_command_buffers": p4a_report.p3a_counters.actual_command_buffers,
                "p3a_gpu_dispatches": p4a_report.p3a_counters.actual_gpu_dispatches,
                "p4a_command_buffers": p4a_report.p4a_counters.actual_command_buffers,
                "p4a_gpu_dispatches": p4a_report.p4a_counters.actual_gpu_dispatches,
                "p4a_authority_receipt_seal_sha256": p4a_report.p4a_source_bindings.p4a_authority_receipt_seal_sha256,
                "p4a_topology_receipt_seal_sha256": p4a_report.p4a_source_bindings.p4a_topology_receipt_seal_sha256,
                "attention_output_device_bytes": p4a_report.attention_output_device_bytes,
                "host_intermediate_handoff_bytes": p4a_report.p3a_counters.host_intermediate_handoff_bytes + p4a_report.p4a_counters.host_intermediate_handoff_bytes,
                "relation": p4a_predecessor_relation,
            },
            "source_controls": {
                "ffn_norm": source_binding_json(&source.ffn_norm),
                "hc_ffn_fn": source_binding_json(&source.hc_ffn_fn),
                "hc_ffn_base": source_binding_json(&source.hc_ffn_base),
                "hc_ffn_scale": source_binding_json(&source.hc_ffn_scale),
                "staging": "live scheduler MhcFfnControl lease, uploaded directly to the borrowed P3A/P4A Metal context",
            },
            "source_gate_reference": &source_gate_reference,
            "p6_residency": {
                "top_slot_route_ids": p6_bindings.selected_expert_ids_top_slot_order,
                "numeric_source_combine_order": numeric_order,
                "hot_capacity_bytes": cache_after_prepare.hot_capacity_bytes,
                "hot_resident_bytes": cache_after_prepare.hot_resident_bytes,
                "cold_resident_bytes": cache_after_prepare.cold_resident_bytes,
                "host_activation_handoff_permitted": p6_bindings.host_activation_handoff_permitted,
                "host_route_weight_handoff_permitted": p6_bindings.host_route_weight_handoff_permitted,
            },
            "p7_p6_graph": {
                "command_buffers": committed_command_buffers,
                "cpu_visible_completion_waits": expected_waits,
                "gpu_dispatches": expected_dispatches,
                "compute_encoders": expected_encoders,
                "physical_trace_command_buffers": physical_counts.command_count,
                "physical_trace_compute_encoders": physical_counts.encoder_count,
                "trace_samples": trace.len(),
                "buffers_created_during_graph": graph_buffers_created,
                "bytes_allocated_during_graph": graph_bytes_allocated,
                "ordered_command_batches": ordered_command_batches(&trace),
            },
            "post_completion_same_input_parity": {
                "post_completion_readbacks_are_not_graph_handoffs": true,
                "mHC_pre_norm": {
                    "reference": mhc_pre_norm_reference,
                    "control_bounds": control_bounds,
                    "storage_bounds": storage_bounds,
                    "flat_rsqrt": mhc_flat_rsqrt_v21,
                    "mixes": mhc_mixes_v21,
                    "pre": mhc_pre_v21,
                    "post": mhc_post_v21,
                    "comb": mhc_comb_v21,
                    "reduced_bf16_store": mhc_reduced_v21,
                    "ffn_norm_source_f32_bit_exact": ffn_norm_source_f32_exact,
                    "pass": mhc_v21_pass,
                },
                "route": {
                    "reference": source_gate_route_reference(
                        source_gate_calibration.as_ref(),
                        source_gate_route_calibration.as_ref(),
                    ),
                    "bounds": route_bounds,
                    "gate_logits": gate_logits_v21,
                    "original_scores": original_scores_v21,
                    "selected_weights": route_weights_v21,
                    "route_ids_source_cpu": same_input_ids,
                    "route_ids_device": route_ids,
                    "route_valid_word": route_valid,
                    "pass": route_v21_pass,
                },
                "direct_upstream_gate_route_parity": &direct_upstream_gate_route_parity,
                "moe_body": {
                    "reference": "same device-produced P7 FFn-norm BF16 row; independent native FP4/FP8 source-F32 and FP64 decode, BF16 stores, SwiGLU and numeric-order combine",
                    "source_numeric_combine_order": source_f32_order,
                    "fp64_numeric_combine_order": f64_order,
                    "storage_bounds": storage_bounds,
                    "moe_output_bf16_store": moe_v21,
                    "pass": moe_v21.pass,
                },
                "mhc_post_child": {
                    "reference": mhc_post_child_reference,
                    "storage_bounds": storage_bounds,
                    "child_hc_state_bf16_store": child_v21,
                    "pass": child_v21.pass,
                },
                "all_scored_sections_pass": all_same_input_v21_pass,
                "observed_hashes": {
                    "p4a_attention_hc_post_bf16": sha256_u16(&attention_hc_post),
                    "p7_ffn_norm_bf16": sha256_u16(&ffn_norm),
                    "p6_moe_output_bf16": sha256_u16(&moe),
                    "p7_child_hc_state_bf16": sha256_u16(&child),
                },
            },
            "claim_boundary": claim_boundary,
        });
        let rendered = serde_json::to_string_pretty(&decimal_strings(unsigned))?;
        write_new_unsealed_diagnostic(&args.out, &rendered)?;
        println!("{rendered}");
        Ok(())
    }

    fn validate_p4a_report(report: &DeepSeekV4P4aContinuationReport) -> ProbeResult<()> {
        if report.phase != DeepSeekV4P4aContinuationPhase::Complete
            || report.p3a_counters.actual_command_buffers != DSV4F_P3A_Q_CHAIN_DISPATCHES
            || report.p3a_counters.actual_gpu_dispatches != DSV4F_P3A_Q_CHAIN_DISPATCHES
            || report.p4a_counters.actual_command_buffers != 1
            || report.p4a_counters.actual_gpu_dispatches != 11
            || report.p3a_counters.host_intermediate_handoff_bytes != 0
            || report.p4a_counters.host_intermediate_handoff_bytes != 0
            || report.p4a_batch_timing.gpu_duration_us.is_none()
        {
            return Err(
                "existing P3A->P4A BOS predecessor is incomplete or has a host handoff".into(),
            );
        }
        Ok(())
    }

    fn stage_p7_source_lease(
        context: &mut DeepSeekV4ExecutionContext,
        prepared: &hawking_core::gravity_deepseek_v4_execution_context::DeepSeekV4PreparedDecodeInput,
        route_set: DeepSeekV4SelectedRouteSet,
    ) -> ProbeResult<DeepSeekV4P7SourceLeasePreparation> {
        let mut preparation = DeepSeekV4P7SourceLeasePreparation::new(context, prepared, 0)?;
        let mut scheduler = DeepSeekV4LayerPreparationScheduler::new(context, 0, route_set)?;
        loop {
            let step = scheduler
                .execute_next(context)?
                .ok_or("scheduler ended before MhcFfnControl")?;
            if step.stage != DeepSeekV4LayerPreparationStage::MhcFfnControl {
                continue;
            }
            let lease = match step.result {
                DeepSeekV4LayerPreparationResult::ControlLease(lease) => lease,
                DeepSeekV4LayerPreparationResult::RoutedExpertAccesses(_) => {
                    return Err("MhcFfnControl did not provide a control lease".into());
                }
            };
            let payload = context.control_arena().get(lease)?;
            let consumption = preparation.bind_mhc_ffn_control(&step, payload)?;
            if consumption.actual_command_buffers != 0
                || consumption.actual_compute_encoders != 0
                || consumption.actual_gpu_dispatches != 0
                || consumption.actual_cpu_visible_waits != 0
                || consumption.host_intermediate_handoff_bytes != 0
            {
                return Err("P7 source preparation reported unexpected device work or host activation handoff".into());
            }
            return Ok(preparation);
        }
    }

    fn required_hot_cache_bytes(
        reader: &DeepSeekV4FullStreamReader,
        route_ids: &[u16],
    ) -> ProbeResult<u64> {
        route_ids.iter().try_fold(0u64, |total, &expert| {
            let descriptor = resolve_expert_bundle(reader, ExpertBundleKey::new(0, expert))?;
            total
                .checked_add(descriptor.payload_bytes)
                .ok_or_else(|| "P6 hot expert capacity overflow".into())
        })
    }

    fn source_binding_json(binding: &DeepSeekV4P7SourceTensorBinding) -> Value {
        json!({
            "name": binding.name,
            "dtype": binding.dtype,
            "shape": binding.shape,
            "bytes": binding.bytes,
            "sha256": binding.sha256,
        })
    }

    /// Receipt-safe provenance for the CPU Gate reference selected by this
    /// diagnostic. The qualified target itself is never serialized here.
    fn source_gate_reference_json(
        calibration: Option<&DeepSeekV4P0GateTorchF32Calibration>,
        route_calibration: Option<&DeepSeekV4P0GateTorchF32RouteCalibration>,
    ) -> ProbeResult<Value> {
        match (route_calibration, calibration) {
            (Some(calibration), None) => {
                let binding = calibration.bindings();
                Ok(json!({
                    "mode": "QUALIFIED_SOURCE_CPU_TORCH_F32_FULL_GATE_ROUTE_TARGET",
                    "selected_for_cpu_route_and_moe_diagnostic": true,
                    "default_serial_rust_oracle_replaced": false,
                    "runtime_or_device_route_changed": false,
                    "post_completion_bf16_gate_input_sha256_verified": true,
                    "direct_upstream_gate_route_parity_eligible": true,
                    "v2_required_for_direct_upstream_gate_route_parity": true,
                    "target_payload_emitted": false,
                    "target_geometry": {
                        "torch_logits": { "dtype": "F32", "shape": [256], "bytes": 1024 },
                        "original_scores": { "dtype": "F32", "shape": [256], "bytes": 1024 },
                        "selected_weights": { "dtype": "F32", "shape": [6], "bytes": 24 },
                        "selected_expert_ids": { "dtype": "U16", "shape": [6], "bytes": 12 },
                        "raw_target_total_bytes": 2084,
                    },
                    "calibration": {
                        "schema": P0_GATE_TORCH_F32_ROUTE_CALIBRATION_SCHEMA,
                        "path": binding.calibration_path.display().to_string(),
                        "file_sha256": binding.calibration_file_sha256,
                        "canonical_sha256": binding.calibration_canonical_sha256,
                        "torch_logits_f32_le_sha256": binding.torch_logits_f32_le_sha256,
                        "original_scores_f32_le_sha256": binding.original_scores_f32_le_sha256,
                        "selected_weights_f32_le_sha256": binding.selected_weights_f32_le_sha256,
                        "selected_expert_ids_u16_le_sha256": binding.selected_expert_ids_u16_le_sha256,
                        "trace_path": binding.trace_path.display().to_string(),
                        "trace_file_sha256": binding.trace_file_sha256,
                        "trace_gate_input_bf16_le_sha256": binding.trace_gate_input_bf16_le_sha256,
                        "trace_baseline_gate_logits_f32_le_sha256": binding.trace_baseline_gate_logits_f32_le_sha256,
                        "artifact_manifest_seal_sha256": binding.artifact_manifest_seal_sha256,
                        "artifact_manifest_file_sha256": binding.artifact_manifest_file_sha256,
                        "artifact_restart_receipt_seal_sha256": binding.artifact_restart_receipt_seal_sha256,
                        "source_repository": binding.source_repository,
                        "source_revision": binding.source_revision,
                        "source_model_py_sha256": binding.source_model_py_sha256,
                        "gate_weight_name": binding.gate_weight_name,
                        "gate_weight_sha256": binding.gate_weight_sha256,
                        "tid2eid_name": binding.tid2eid_name,
                        "tid2eid_sha256": binding.tid2eid_sha256,
                        "route_scale": binding.route_scale,
                        "layer": binding.layer,
                        "token_id": binding.token_id,
                        "token_position": binding.token_position,
                    },
                    "direct_route_runtime_metadata": direct_route_runtime_metadata()?,
                    "claim_boundary": "This v2 opt-in target affects only the post-completion CPU route/MoE diagnostic after its exact BF16 Gate-input hash has matched. It does not change the real graph and any V2.1 result remains a bounded P0 diagnostic, not a runtime, endpoint, generation, or TPS result.",
                }))
            }
            (None, Some(calibration)) => {
                let binding = calibration.bindings();
                Ok(json!({
                    "mode": "QUALIFIED_SOURCE_CPU_TORCH_F32_GATE_LOGIT_TARGET_V1_INCOMPLETE_FOR_DIRECT_ROUTE_PARITY",
                    "selected_for_cpu_route_and_moe_diagnostic": true,
                    "default_serial_rust_oracle_replaced": false,
                    "runtime_or_device_route_changed": false,
                    "post_completion_bf16_gate_input_sha256_verified": true,
                    "direct_upstream_gate_route_parity_eligible": false,
                    "v2_required_for_direct_upstream_gate_route_parity": true,
                    "incomplete_reason": "v1 retains F32[256] Torch logits only; its sqrt-softplus, tid2eid gather, normalization, and route-scale values are recomputed by the diagnostic rather than directly source-targeted.",
                    "target_payload_emitted": false,
                    "target_geometry": { "dtype": "F32", "shape": [256], "bytes": 1024 },
                    "calibration": {
                        "path": binding.calibration_path.display().to_string(),
                        "file_sha256": binding.calibration_file_sha256,
                        "torch_logits_f32_le_sha256": binding.torch_logits_f32_le_sha256,
                        "trace_path": binding.trace_path.display().to_string(),
                        "trace_file_sha256": binding.trace_file_sha256,
                        "trace_gate_input_bf16_le_sha256": binding.trace_gate_input_bf16_le_sha256,
                        "trace_baseline_gate_logits_f32_le_sha256": binding.trace_baseline_gate_logits_f32_le_sha256,
                        "artifact_manifest_seal_sha256": binding.artifact_manifest_seal_sha256,
                        "artifact_manifest_file_sha256": binding.artifact_manifest_file_sha256,
                        "artifact_restart_receipt_seal_sha256": binding.artifact_restart_receipt_seal_sha256,
                        "source_repository": binding.source_repository,
                        "source_revision": binding.source_revision,
                        "source_model_py_sha256": binding.source_model_py_sha256,
                        "gate_weight_name": binding.gate_weight_name,
                        "gate_weight_sha256": binding.gate_weight_sha256,
                        "tid2eid_name": binding.tid2eid_name,
                        "tid2eid_sha256": binding.tid2eid_sha256,
                        "layer": binding.layer,
                        "token_id": binding.token_id,
                        "token_position": binding.token_position,
                    },
                    "claim_boundary": "This v1 opt-in target affects only the post-completion CPU route/MoE diagnostic after its exact BF16 Gate-input hash has matched. It neither changes the real graph nor establishes a V2.1 pass, runtime, endpoint, generation, or TPS result by itself.",
                }))
            }
            (None, None) => Ok(json!({
                "mode": "SERIAL_SOURCE_DERIVED_RUST_GATE_DIAGNOSTIC_TRANSCRIPTION",
                "selected_for_cpu_route_and_moe_diagnostic": true,
                "default_serial_rust_oracle_replaced": false,
                "runtime_or_device_route_changed": false,
                "post_completion_bf16_gate_input_sha256_verified": false,
                "direct_upstream_gate_route_parity_eligible": false,
                "v2_required_for_direct_upstream_gate_route_parity": true,
                "upstream_operator": "inference/model.py framework F.linear",
                "framework_instruction_or_reduction_order_claim": false,
                "claim_boundary": "The default Rust loop is a serial source-derived diagnostic/transcription over admitted BF16 rows, not an assertion of exact upstream framework F.linear arithmetic. It neither changes the real graph nor establishes a V2.1 pass, runtime, endpoint, generation, or TPS result by itself.",
            })),
            (Some(_), Some(_)) => {
                Err("v1 and v2 Gate calibrations must remain mutually exclusive".into())
            }
        }
    }

    fn direct_route_runtime_metadata() -> ProbeResult<Value> {
        Ok(json!({
            "executable": executable_provenance()?,
            "build_target": {
                "architecture": std::env::consts::ARCH,
                "operating_system": std::env::consts::OS,
                "family": std::env::consts::FAMILY,
                "debug_assertions": cfg!(debug_assertions),
            },
            "p6_c4_gate": {
                "kernel": P6_C4_GATE_KERNEL,
                "grid_threads": [P6_C4_GATE_GRID_THREADS, 1, 1],
                "threads_per_threadgroup": [P6_C4_GATE_SIMDGROUP_THREADS, 1, 1],
            },
        }))
    }

    fn source_gate_cpu_semantics(
        calibration: Option<&DeepSeekV4P0GateTorchF32Calibration>,
        route_calibration: Option<&DeepSeekV4P0GateTorchF32RouteCalibration>,
    ) -> Value {
        match (route_calibration, calibration) {
            (Some(_), None) => json!({
                "kind": "qualified_torch_f32_external_full_gate_route_target_v2",
                "details": "Bound, artifact- and trace-verified Torch F32 logits, sqrt-softplus original scores, tid2eid-selected IDs, normalized route weights, and route scale feed only the post-completion CPU MoE diagnostic after the completed BF16[4096] Gate-input SHA-256 matches.",
            }),
            (None, Some(_)) => json!({
                "kind": "qualified_torch_f32_external_logit_target_v1_incomplete_for_direct_route_parity",
                "details": "Bound, artifact- and trace-verified F32[256] Torch Gate logits feed only the post-completion CPU sqrt-softplus/hash route/MoE diagnostic after the completed BF16[4096] Gate-input SHA-256 matches. v1 does not directly target route IDs or weights.",
            }),
            (None, None) => json!({
                "kind": "serial_source_derived_rust_transcription",
                "details": "Rust layer0_hash_route_cpu_oracle_for_token reads one row-major BF16 Gate row at a time in increasing columns 0..4095 with `accumulator += activation * weight`; this records a source-derived diagnostic order, not an assertion about upstream framework F.linear instruction or reduction behavior.",
            }),
            (Some(_), Some(_)) => json!({
                "kind": "invalid_mutually_exclusive_v1_v2_calibrations",
            }),
        }
    }

    fn source_gate_route_reference(
        calibration: Option<&DeepSeekV4P0GateTorchF32Calibration>,
        route_calibration: Option<&DeepSeekV4P0GateTorchF32RouteCalibration>,
    ) -> &'static str {
        match (route_calibration, calibration) {
            (Some(_), None) => "same device-produced P7 FFn-norm BF16 row; qualified v2 direct Torch F32 Gate logits/sqrt-softplus/tid2eid/normalized-route target bound to the completed row, plus independent FP64 Gate/sqrt-softplus/hash-route authority",
            (None, Some(_)) => "same device-produced P7 FFn-norm BF16 row; qualified v1 Torch F32 Gate-logit target bound to the completed row while post-linear routing is recomputed by the diagnostic, plus independent FP64 Gate/sqrt-softplus/hash-route authority",
            (None, None) => "same device-produced P7 FFn-norm BF16 row; serial source-derived Rust F32 Gate diagnostic/transcription plus independent FP64 Gate/sqrt-softplus/hash-route authority",
            (Some(_), Some(_)) => "invalid mutually exclusive v1/v2 Gate calibration selection",
        }
    }

    fn direct_upstream_gate_route_parity_json(
        route_calibration: Option<&DeepSeekV4P0GateTorchF32RouteCalibration>,
        gate_logits: &PairedScore,
        original_scores: &PairedScore,
        selected_weights: &PairedScore,
        selected_ids_exact_device_match: bool,
        route_valid: bool,
    ) -> Value {
        match route_calibration {
            Some(calibration) => {
                let binding = calibration.bindings();
                let pass = gate_logits.pass
                    && original_scores.pass
                    && selected_weights.pass
                    && selected_ids_exact_device_match
                    && route_valid;
                json!({
                    "status": "QUALIFIED_V2_DIRECT_SOURCE_CPU_TORCH_GATE_ROUTE_COMPARISON_NUMERIC_PARITY_V2_1_ONLY",
                    "claimed": true,
                    "v2_calibration_required": true,
                    "raw_target_payload_emitted": false,
                    "source_operator_path": "F.linear -> F.softplus -> sqrt -> tid2eid gather -> normalize -> route_scale",
                    "calibration": {
                        "schema": P0_GATE_TORCH_F32_ROUTE_CALIBRATION_SCHEMA,
                        "file_sha256": binding.calibration_file_sha256,
                        "canonical_sha256": binding.calibration_canonical_sha256,
                        "torch_logits_f32_le_sha256": binding.torch_logits_f32_le_sha256,
                        "original_scores_f32_le_sha256": binding.original_scores_f32_le_sha256,
                        "selected_weights_f32_le_sha256": binding.selected_weights_f32_le_sha256,
                        "selected_expert_ids_u16_le_sha256": binding.selected_expert_ids_u16_le_sha256,
                    },
                    "bounded_scores": {
                        "gate_logits": gate_logits,
                        "original_scores": original_scores,
                        "selected_weights": selected_weights,
                        "selected_ids_exact_device_match": selected_ids_exact_device_match,
                        "route_valid_word": route_valid,
                    },
                    "pass": pass,
                    "claim_boundary": "A pass is direct v2 source-CPU-Torch versus completed-device P0 Gate-route numeric comparison only. It neither changes the graph nor establishes a runtime, endpoint, generation, or TPS result.",
                })
            }
            None => json!({
                "status": "NOT_CLAIMED_V2_ROUTE_CALIBRATION_REQUIRED_V1_LOGIT_ONLY_OR_SERIAL_DIAGNOSTIC",
                "claimed": false,
                "v2_calibration_required": true,
                "raw_target_payload_emitted": false,
                "reason": "Direct upstream Gate-route parity is intentionally withheld unless the mutually exclusive v2 bounded route calibration is loaded.",
            }),
        }
    }

    /// Emit one deliberately bounded raw activation trace for Gate arithmetic
    /// diagnostics. This is a post-completion observation only: it is neither
    /// consumed by the graph nor eligible to become a parity receipt.
    fn write_new_gate_input_trace(
        path: &Path,
        reader: &DeepSeekV4FullStreamReader,
        source: &hawking_core::gravity_deepseek_v4_p7_composition::DeepSeekV4P7FfnSourceContract,
        p6_bindings: &hawking_core::gravity_deepseek_v4_p6_device::DeepSeekV4P6SourceBindings,
        p4a_attention_hc_post: &[u16],
        ffn_norm: &[u16],
        gate_logits: &[f32],
        metal_device_name: &str,
        compile_math_mode: &str,
        strict_math: bool,
        p4a_predecessor_exact: bool,
        p4a_predecessor_label: &str,
    ) -> ProbeResult<Value> {
        // Recheck immediately before the create-new publish to make an
        // already-present path a deterministic refusal. The hard-link publish
        // below closes the remaining check-to-create race without overwrite.
        validate_new_gate_input_trace_path(path)?;
        if source.layer != 0
            || source.token_id != PREFIX_TOKEN_ID as u32
            || source.token_position != 0
            || p6_bindings.layer != source.layer
            || p6_bindings.token_id != source.token_id
            || p6_bindings.token_position != source.token_position
        {
            return Err("Gate input trace is not bound to the P0 BOS source contract".into());
        }
        if p4a_attention_hc_post.len() != HC_FLAT_WIDTH
            || ffn_norm.len() != HIDDEN_SIZE
            || gate_logits.len() != 256
        {
            return Err("Gate input trace received an unexpected P0 tensor geometry".into());
        }
        if gate_logits.iter().any(|value| !value.is_finite()) {
            return Err("Gate input trace received non-finite Gate logits".into());
        }

        let raw_payload = bf16_le_bytes(ffn_norm);
        if raw_payload.len() != GATE_INPUT_TRACE_RAW_PAYLOAD_MAX_BYTES {
            return Err("Gate input trace raw payload exceeded its fixed BF16[4096] bound".into());
        }
        let raw_payload_sha256 = sha256(&raw_payload);
        let raw_payload_hex = lowercase_hex(&raw_payload);
        if raw_payload_hex.len() != raw_payload.len() * 2 {
            return Err("Gate input trace hex encoding length is invalid".into());
        }
        let raw_payload_hex_bytes = raw_payload_hex.len();
        let p4a_input_sha256 = sha256_u16(p4a_attention_hc_post);
        let gate_output_sha256 = sha256_f32(gate_logits);

        let trace = json!({
            "schema": GATE_INPUT_TRACE_SCHEMA,
            "status": GATE_INPUT_TRACE_STATUS,
            "unsealed": true,
            "receipt_promoted": false,
            "is_receipt": false,
            "claim_boundary": "This is one post-completion BOS Gate-input observation for arithmetic diagnosis only. It proves neither a Numeric Parity V2.1 pass nor a P7 exact-storage, causal runtime, 43-layer runtime, HCLI, token-generation, TPS, or capability result.",
            "trace_binding": {
                "layer": source.layer,
                "token_id": source.token_id,
                "token_position": source.token_position,
                "post_completion_readback_only": true,
                "real_graph_completed_before_trace_emission": true,
                "trace_does_not_feed_graph": true,
                "trace_does_not_modify_graph_counters": true,
                "p4a_attention_predecessor_exact": p4a_predecessor_exact,
                "p4a_attention_predecessor_label": p4a_predecessor_label,
            },
            "artifact": {
                "path": reader.artifact_root().display().to_string(),
                "manifest_seal_sha256": reader.manifest_seal_sha256(),
                "manifest_file_sha256": reader.manifest_file_sha256(),
                "restart_receipt_seal_sha256": reader.restart_seal_sha256(),
                "source_parent_retained": false,
            },
            "model_source": {
                "repository": reader.source_identity().repository,
                "revision": reader.source_identity().revision,
                "metadata_asset_sha256": {
                    "inference/model.py": reader.source_metadata_asset_sha256("inference/model.py")?,
                    "inference/kernel.py": reader.source_metadata_asset_sha256("inference/kernel.py")?,
                    "inference/convert.py": reader.source_metadata_asset_sha256("inference/convert.py")?,
                    "inference/config.json": reader.source_metadata_asset_sha256("inference/config.json")?,
                    "config.json": reader.source_metadata_asset_sha256("config.json")?,
                },
                "p7_static_control_bindings": {
                    "ffn_norm": source_binding_json(&source.ffn_norm),
                    "hc_ffn_fn": source_binding_json(&source.hc_ffn_fn),
                    "hc_ffn_base": source_binding_json(&source.hc_ffn_base),
                    "hc_ffn_scale": source_binding_json(&source.hc_ffn_scale),
                },
                "p6_gate_route_bindings": {
                    "gate_weight_name": &p6_bindings.gate_weight_name,
                    "gate_weight_sha256": &p6_bindings.gate_weight_sha256,
                    "tid2eid_name": match &p6_bindings.route {
                        hawking_core::gravity_deepseek_v4_p6_device::DeepSeekV4P6GateRouteBinding::HashTid2Eid {
                            tid2eid_name,
                            ..
                        } => tid2eid_name.as_str(),
                        hawking_core::gravity_deepseek_v4_p6_device::DeepSeekV4P6GateRouteBinding::LearnedBias {
                            bias_name,
                            ..
                        } => bias_name.as_str(),
                    },
                    "tid2eid_sha256": match &p6_bindings.route {
                        hawking_core::gravity_deepseek_v4_p6_device::DeepSeekV4P6GateRouteBinding::HashTid2Eid {
                            tid2eid_sha256,
                            ..
                        } => tid2eid_sha256.as_str(),
                        hawking_core::gravity_deepseek_v4_p6_device::DeepSeekV4P6GateRouteBinding::LearnedBias {
                            bias_sha256,
                            ..
                        } => bias_sha256.as_str(),
                    },
                    "selected_expert_ids_top_slot_order": &p6_bindings.selected_expert_ids_top_slot_order,
                },
            },
            "executable": {
                "runner": executable_provenance()?,
                "metal_device": metal_device_name,
                "compile_math_mode": compile_math_mode,
                "strict_math_candidate": strict_math,
                "graph_execution": "P3A/P4A -> P7 -> P6 -> P7 completed before every trace readback and file write",
            },
            "input_output_sha256": {
                "p7_producer_input_p4a_attention_hc_post_bf16_le": &p4a_input_sha256,
                "p7_producer_output_ffn_norm_bf16_le": &raw_payload_sha256,
                "p6_gate_input_ffn_norm_bf16_le": &raw_payload_sha256,
                "p6_gate_output_logits_f32_le": &gate_output_sha256,
            },
            "raw_payload": {
                "name": "p7_ffn_norm_bf16_bos_gate_input",
                "role": "the sole raw payload; P7 producer output and P6 Gate input",
                "shape": [HIDDEN_SIZE],
                "dtype": "BF16",
                "byte_order": "little_endian",
                "decode_contract": "data has exactly 2 * element_count lowercase ASCII hex characters. Decode adjacent byte pairs in increasing element order, then use u16::from_le_bytes; the resulting u16 bits are the BF16 values.",
                "element_count": HIDDEN_SIZE,
                "byte_count": raw_payload.len(),
                "sha256": &raw_payload_sha256,
                "encoding": "lowercase_hex_raw_bf16_le",
                "data": &raw_payload_hex,
            },
            "privacy_and_storage_bound": {
                "raw_payload_count": 1,
                "raw_payload_allowlist": ["p7_ffn_norm_bf16_bos_gate_input"],
                "raw_payload_hard_max_bytes": GATE_INPUT_TRACE_RAW_PAYLOAD_MAX_BYTES,
                "raw_payload_actual_bytes": raw_payload.len(),
                "raw_payload_encoded_hex_bytes": raw_payload_hex_bytes,
                "serialized_trace_hard_max_bytes": GATE_INPUT_TRACE_SERIALIZED_MAX_BYTES,
                "raw_source_weight_payloads": 0,
                "raw_other_activation_payloads": 0,
                "raw_gate_output_payloads": 0,
                "raw_route_weight_payloads": 0,
                "policy": "Bounded local diagnostic data only: exactly one BOS BF16[4096] Gate-input row is retained. All source weights, other activations, Gate logits, routes, and later-token state are represented only by metadata or SHA-256.",
            },
            "publication": {
                "path": path.display().to_string(),
                "mode": "create_new_only_atomic_hard_link",
                "normal_out_independent": true,
                "normal_out_is_not_written_when_v2_1_fails": true,
            },
        });
        let rendered = serde_json::to_string_pretty(&decimal_strings(trace))?;
        if rendered.len() > GATE_INPUT_TRACE_SERIALIZED_MAX_BYTES {
            return Err("Gate input trace exceeded its fixed serialized storage bound".into());
        }
        write_new_unsealed_diagnostic(path, &rendered)?;
        Ok(json!({
            "requested": true,
            "written": true,
            "schema": GATE_INPUT_TRACE_SCHEMA,
            "status": GATE_INPUT_TRACE_STATUS,
            "unsealed": true,
            "receipt_promoted": false,
            "path": path.display().to_string(),
            "raw_payload_count": 1,
            "raw_payload_bytes": GATE_INPUT_TRACE_RAW_PAYLOAD_MAX_BYTES,
            "raw_payload_sha256": raw_payload_sha256,
        }))
    }

    fn executable_provenance() -> ProbeResult<Value> {
        let requested_path = std::env::current_exe()?;
        if fs::symlink_metadata(&requested_path)?
            .file_type()
            .is_symlink()
        {
            return Err(
                "current executable symlink is not admitted for Gate trace provenance".into(),
            );
        }
        let path = fs::canonicalize(requested_path)?;
        let metadata = fs::metadata(&path)?;
        if !metadata.is_file() || metadata.len() == 0 {
            return Err(
                "current executable is not a nonempty regular file for Gate trace provenance"
                    .into(),
            );
        }
        let build_profile_hint = if path
            .components()
            .any(|component| component.as_os_str() == "release")
        {
            "release"
        } else if path
            .components()
            .any(|component| component.as_os_str() == "debug")
        {
            "debug"
        } else {
            "unknown"
        };
        Ok(json!({
            "path": path.display().to_string(),
            "sha256": sha256(&fs::read(&path)?),
            "bytes": metadata.len(),
            "build_profile_hint": build_profile_hint,
            "cargo_package": env!("CARGO_PKG_NAME"),
            "cargo_package_version": env!("CARGO_PKG_VERSION"),
        }))
    }

    fn bf16_le_bytes(values: &[u16]) -> Vec<u8> {
        values
            .iter()
            .flat_map(|value| value.to_le_bytes())
            .collect()
    }

    fn sha256_f32(values: &[f32]) -> String {
        let bytes = values
            .iter()
            .flat_map(|value| value.to_bits().to_le_bytes())
            .collect::<Vec<_>>();
        sha256(&bytes)
    }

    fn lowercase_hex(bytes: &[u8]) -> String {
        const HEX: &[u8; 16] = b"0123456789abcdef";
        let mut encoded = String::with_capacity(bytes.len() * 2);
        for &byte in bytes {
            encoded.push(char::from(HEX[usize::from(byte >> 4)]));
            encoded.push(char::from(HEX[usize::from(byte & 0x0f)]));
        }
        encoded
    }

    fn ids_u32(ids: &[u64]) -> ProbeResult<Vec<u32>> {
        ids.iter()
            .map(|&id| u32::try_from(id).map_err(|_| "source route ID exceeds u32".into()))
            .collect()
    }

    fn source_combine_order_u32(order: &[Layer0MoeCombineOrder]) -> ProbeResult<Vec<(u32, u32)>> {
        order
            .iter()
            .map(|entry| {
                Ok((
                    u32::try_from(entry.source_top_slot)
                        .map_err(|_| "source MoE top slot exceeds u32")?,
                    u32::try_from(entry.expert_id)
                        .map_err(|_| "source MoE expert ID exceeds u32")?,
                ))
            })
            .collect()
    }

    fn read_gpu_bytes(buffer: &metal::Buffer, length: usize) -> ProbeResult<Vec<u8>> {
        if buffer.length() < length as u64 {
            return Err("Metal buffer is smaller than post-completion diagnostic readback".into());
        }
        Ok(unsafe { std::slice::from_raw_parts(buffer.contents() as *const u8, length).to_vec() })
    }

    fn read_gpu_u16(buffer: &metal::Buffer, count: usize) -> ProbeResult<Vec<u16>> {
        Ok(read_gpu_bytes(buffer, count * std::mem::size_of::<u16>())?
            .chunks_exact(2)
            .map(|chunk| u16::from_le_bytes([chunk[0], chunk[1]]))
            .collect())
    }

    fn read_gpu_u32(buffer: &metal::Buffer, count: usize) -> ProbeResult<Vec<u32>> {
        Ok(read_gpu_bytes(buffer, count * std::mem::size_of::<u32>())?
            .chunks_exact(4)
            .map(|chunk| u32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]))
            .collect())
    }

    fn read_gpu_f32(buffer: &metal::Buffer, count: usize) -> ProbeResult<Vec<f32>> {
        let values = read_gpu_bytes(buffer, count * std::mem::size_of::<f32>())?
            .chunks_exact(4)
            .map(|chunk| {
                f32::from_bits(u32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]))
            })
            .collect::<Vec<_>>();
        if values.iter().any(|value| !value.is_finite()) {
            return Err("device diagnostic F32 buffer contains a non-finite value".into());
        }
        Ok(values)
    }

    fn p7_mhc_control_v21_bounds() -> Bounds {
        Bounds {
            max_abs_near_zero: 1.0e-3,
            max_relative_l2: 2.0e-4,
            min_cosine: 0.999_999,
            max_kl: 0.0,
            require_kl: false,
            top_k: 5,
            max_meaningful_rel: 5.0e-2,
            gate_max_meaningful_rel: true,
        }
    }

    fn p7_bf16_storage_v21_bounds() -> Bounds {
        Bounds {
            max_abs_near_zero: 2.0e-2,
            max_relative_l2: 1.0e-2,
            min_cosine: 0.999,
            max_kl: 0.0,
            require_kl: false,
            top_k: 5,
            max_meaningful_rel: 1.0e-1,
            gate_max_meaningful_rel: true,
        }
    }

    fn bf16_bits_f32(bits: &[u16]) -> Vec<f32> {
        bits.iter()
            .map(|&value| f32::from_bits(u32::from(value) << 16))
            .collect()
    }

    fn bf16_bits_f64(bits: &[u16]) -> Vec<f64> {
        bits.iter()
            .map(|&value| f64::from(f32::from_bits(u32::from(value) << 16)))
            .collect()
    }

    /// Keep a failed P7 pre/norm store diagnosis bounded: it records the
    /// total disagreement plus at most eight BF16 lanes, never a hidden-state
    /// payload. This report is emitted only to stderr on a V2.1 failure and
    /// is never sent to the artifact writer.
    fn bf16_store_mismatch_summary(source_f32: &[u16], device: &[u16]) -> Value {
        const MAX_SAMPLES: usize = 8;
        let common = source_f32.len().min(device.len());
        let mut mismatched_elements = source_f32.len().abs_diff(device.len());
        let mut first_mismatches = Vec::new();
        for index in 0..common {
            if source_f32[index] != device[index] {
                mismatched_elements += 1;
                if first_mismatches.len() < MAX_SAMPLES {
                    first_mismatches.push(json!({
                        "index": index,
                        "source_f32_bf16_bits": format!("0x{:04x}", source_f32[index]),
                        "device_bf16_bits": format!("0x{:04x}", device[index]),
                    }));
                }
            }
        }
        json!({
            "source_f32_bit_exact": source_f32 == device,
            "source_f32_elements": source_f32.len(),
            "device_elements": device.len(),
            "mismatched_elements": mismatched_elements,
            "first_mismatches_bounded": first_mismatches,
        })
    }

    /// Report the small set of Gate rows that actually control a V2.1
    /// failure. This is deliberately based on completed scalar logits only:
    /// no Gate weights or FFn activation row is retained or emitted.
    fn bounded_gate_logit_mismatch_diagnostics(
        source_f32: &[f32],
        device_f32: &[f32],
        reference_f64: &[f64],
        meaningful_cutoff: f64,
    ) -> ProbeResult<Value> {
        const MAX_SAMPLES: usize = 8;
        if source_f32.len() != device_f32.len()
            || source_f32.len() != reference_f64.len()
            || !meaningful_cutoff.is_finite()
            || meaningful_cutoff < 0.0
        {
            return Err("Gate mismatch diagnostic received incompatible vectors or cutoff".into());
        }
        if source_f32.iter().any(|value| !value.is_finite())
            || device_f32.iter().any(|value| !value.is_finite())
            || reference_f64.iter().any(|value| !value.is_finite())
        {
            return Err("Gate mismatch diagnostic received a non-finite score".into());
        }

        // Match the failed device-side V2.1 diagnostic's most informative
        // per-lane signal: error against the independent FP64 authority,
        // ordered by meaningful-scale relative error, then absolute error.
        let mut lanes: Vec<(usize, f64, f64)> = reference_f64
            .iter()
            .enumerate()
            .map(|(index, &reference)| {
                let absolute = (f64::from(device_f32[index]) - reference).abs();
                let relative = if reference.abs() >= meaningful_cutoff {
                    absolute / reference.abs()
                } else {
                    0.0
                };
                (index, relative, absolute)
            })
            .collect();
        lanes.sort_by(|left, right| {
            right
                .1
                .total_cmp(&left.1)
                .then_with(|| right.2.total_cmp(&left.2))
                .then_with(|| left.0.cmp(&right.0))
        });

        let samples = lanes
            .into_iter()
            .take(MAX_SAMPLES)
            .map(|(index, _, _)| {
                let source = source_f32[index];
                let device = device_f32[index];
                let reference = reference_f64[index];
                let source_abs = (f64::from(source) - reference).abs();
                let device_abs = (f64::from(device) - reference).abs();
                let source_device_abs = (f64::from(source) - f64::from(device)).abs();
                let denominator = reference.abs().max(f64::MIN_POSITIVE);
                let source_device_denominator = f64::from(source)
                    .abs()
                    .max(f64::from(device).abs())
                    .max(f64::MIN_POSITIVE);
                let rounded_reference = reference as f32;
                json!({
                    "row": index,
                    "meaningful_scale": reference.abs() >= meaningful_cutoff,
                    "source_f32": source,
                    "device_f32": device,
                    "fp64_authority": reference,
                    "source_f32_bits": format!("0x{:08x}", source.to_bits()),
                    "device_f32_bits": format!("0x{:08x}", device.to_bits()),
                    "fp64_rounded_to_f32_bits": format!("0x{:08x}", rounded_reference.to_bits()),
                    "source_vs_fp64_abs": source_abs,
                    "source_vs_fp64_rel": source_abs / denominator,
                    "device_vs_fp64_abs": device_abs,
                    "device_vs_fp64_rel": device_abs / denominator,
                    "source_vs_device_abs": source_device_abs,
                    "source_vs_device_rel": source_device_abs / source_device_denominator,
                    "source_device_ulp": f32_ulp_distance(source, device),
                    "source_to_fp64_rounded_ulp": f32_ulp_distance(source, rounded_reference),
                    "device_to_fp64_rounded_ulp": f32_ulp_distance(device, rounded_reference),
                })
            })
            .collect::<Vec<_>>();
        Ok(json!({
            "ranking": "descending device_f32-vs-FP64 meaningful-scale relative error, then absolute error, then row",
            "meaningful_scale_cutoff": meaningful_cutoff,
            "candidate_rows": source_f32.len(),
            "maximum_rows_emitted": MAX_SAMPLES,
            "rows": samples,
        }))
    }

    fn f32_ulp_distance(left: f32, right: f32) -> u64 {
        let ordered = |value: f32| {
            let bits = value.to_bits();
            if bits & 0x8000_0000 != 0 {
                (!bits) as i64
            } else {
                (bits | 0x8000_0000) as i64
            }
        };
        ordered(left).abs_diff(ordered(right))
    }

    fn sha256_u16(values: &[u16]) -> String {
        let bytes = values
            .iter()
            .flat_map(|value| value.to_le_bytes())
            .collect::<Vec<_>>();
        sha256(&bytes)
    }

    fn ordered_command_batches(trace: &[hawking_core::metal::DispatchSample]) -> Vec<Value> {
        const STAGES: [&str; 4] = [
            "P7 mHC-FFN pre plus FFn RMSNorm",
            "P6 Gate/route/W1-W3/cast/SwiGLU",
            "P6 down-QAT/W2/cast/source-order combine",
            "P7 mHC-FFN post",
        ];
        trace
            .iter()
            .enumerate()
            .map(|(index, sample)| {
                json!({
                    "ordinal": index,
                    "stage": STAGES.get(index).copied().unwrap_or("unexpected command batch"),
                    "kernel_name": sample.kernel_name,
                    "host_wall_us": sample.wall_us,
                    "gpu_duration_us": sample.gpu_us,
                    "gpu_start_ns": sample.gpu_start_ns,
                    "gpu_end_ns": sample.gpu_end_ns,
                })
            })
            .collect()
    }

    fn parse_args() -> ProbeResult<Args> {
        let mut artifact = None;
        let mut out = None;
        let mut gate_input_trace_out = None;
        let mut source_gate_calibration = None;
        let mut source_gate_route_calibration = None;
        let mut strict_math = false;
        let mut args = std::env::args_os().skip(1);
        while let Some(flag) = args.next() {
            match flag.to_string_lossy().as_ref() {
                "--artifact" => artifact = args.next().map(PathBuf::from),
                "--out" => out = args.next().map(PathBuf::from),
                "--gate-input-trace-out" => {
                    if gate_input_trace_out.is_some() {
                        return Err("--gate-input-trace-out may be supplied at most once".into());
                    }
                    gate_input_trace_out = Some(PathBuf::from(
                        args.next()
                            .ok_or("--gate-input-trace-out requires an absolute new JSON path")?,
                    ));
                }
                "--source-gate-calibration" => {
                    if source_gate_calibration.is_some() {
                        return Err("--source-gate-calibration may be supplied at most once".into());
                    }
                    let path = PathBuf::from(args.next().ok_or(
                        "--source-gate-calibration requires an absolute qualified calibration JSON path",
                    )?);
                    if !path.is_absolute() {
                        return Err("--source-gate-calibration must be an absolute path".into());
                    }
                    source_gate_calibration = Some(path);
                }
                "--source-gate-route-calibration" => {
                    if source_gate_route_calibration.is_some() {
                        return Err(
                            "--source-gate-route-calibration may be supplied at most once".into(),
                        );
                    }
                    let path = PathBuf::from(args.next().ok_or(
                        "--source-gate-route-calibration requires an absolute qualified v2 route calibration JSON path",
                    )?);
                    if !path.is_absolute() {
                        return Err(
                            "--source-gate-route-calibration must be an absolute path".into()
                        );
                    }
                    source_gate_route_calibration = Some(path);
                }
                "--strict-math" => strict_math = true,
                "--help" | "-h" => {
                    println!(
                        "usage: gravity_deepseek_v4_layer0_position0_p7_device --artifact <absolute full Gravity dir> --out <absolute new unsealed diagnostic.json> [--gate-input-trace-out <absolute new unsealed non-receipt JSON>] [--source-gate-calibration <absolute qualified v1 calibration JSON> | --source-gate-route-calibration <absolute qualified v2 route calibration JSON>] [--strict-math]"
                    );
                    std::process::exit(0);
                }
                other => return Err(format!("unknown argument {other}").into()),
            }
        }
        let artifact = artifact.ok_or("--artifact required")?;
        let out = out.ok_or("--out required")?;
        if !artifact.is_absolute() || !out.is_absolute() {
            return Err("--artifact and --out must be absolute paths".into());
        }
        if source_gate_calibration.is_some() && source_gate_route_calibration.is_some() {
            return Err(
                "--source-gate-calibration and --source-gate-route-calibration are mutually exclusive"
                    .into(),
            );
        }
        if let Some(path) = gate_input_trace_out.as_ref() {
            if path == &out {
                return Err(
                    "--gate-input-trace-out must differ from --out so a V2.1 failure cannot publish the normal diagnostic path".into(),
                );
            }
            validate_new_gate_input_trace_path(path)?;
        }
        Ok(Args {
            artifact,
            out,
            gate_input_trace_out,
            source_gate_calibration,
            source_gate_route_calibration,
            strict_math,
        })
    }

    fn validate_new_gate_input_trace_path(path: &Path) -> ProbeResult<()> {
        if !path.is_absolute() {
            return Err("--gate-input-trace-out must be an absolute path".into());
        }
        let parent = path
            .parent()
            .filter(|parent| !parent.as_os_str().is_empty())
            .ok_or("--gate-input-trace-out requires a parent directory")?;
        let name = path
            .file_name()
            .and_then(|name| name.to_str())
            .filter(|name| !name.is_empty())
            .ok_or("--gate-input-trace-out filename is not UTF-8")?;
        if name == "." || name == ".." || parent.as_os_str().is_empty() {
            return Err("--gate-input-trace-out must name a new JSON file".into());
        }
        match fs::symlink_metadata(path) {
            Ok(_) => {
                return Err(format!(
                    "refusing to overwrite existing Gate input trace {}",
                    path.display()
                )
                .into())
            }
            Err(error) if error.kind() == std::io::ErrorKind::NotFound => {}
            Err(error) => return Err(Box::new(error)),
        }
        Ok(())
    }

    fn write_new_unsealed_diagnostic(path: &Path, rendered: &str) -> ProbeResult<()> {
        if path.exists() {
            return Err(format!(
                "refusing to overwrite unsealed diagnostic {}",
                path.display()
            )
            .into());
        }
        let parent = path
            .parent()
            .filter(|parent| !parent.as_os_str().is_empty())
            .ok_or("unsealed diagnostic output requires a parent directory")?;
        fs::create_dir_all(parent)?;
        let name = path
            .file_name()
            .and_then(|name| name.to_str())
            .ok_or("unsealed diagnostic output filename is not UTF-8")?;
        let temporary = parent.join(format!(".{name}.{}.p0-p7.tmp", std::process::id()));
        let mut file = OpenOptions::new()
            .write(true)
            .create_new(true)
            .open(&temporary)?;
        if let Err(error) = file
            .write_all(rendered.as_bytes())
            .and_then(|_| file.write_all(b"\n"))
            .and_then(|_| file.sync_all())
        {
            let _ = fs::remove_file(&temporary);
            return Err(Box::new(error));
        }
        drop(file);
        if let Err(error) = fs::hard_link(&temporary, path) {
            let _ = fs::remove_file(&temporary);
            return Err(format!("atomically publish unsealed diagnostic: {error}").into());
        }
        fs::remove_file(&temporary)?;
        File::open(parent)?.sync_all()?;
        Ok(())
    }

    fn decimal_strings(value: Value) -> Value {
        match value {
            Value::Number(number) if number.is_i64() || number.is_u64() => Value::Number(number),
            Value::Number(number) => Value::String(number.to_string()),
            Value::Array(values) => Value::Array(values.into_iter().map(decimal_strings).collect()),
            Value::Object(values) => Value::Object(
                values
                    .into_iter()
                    .map(|(key, value)| (key, decimal_strings(value)))
                    .collect(),
            ),
            other => other,
        }
    }

    fn sha256(bytes: &[u8]) -> String {
        format!("{:x}", Sha256::digest(bytes))
    }
}

#[cfg(target_os = "macos")]
fn main() -> Result<(), Box<dyn std::error::Error>> {
    macos::run()
}

//! Unsealed, bounded all-device DeepSeek-V4 P4B -> P7 -> P6 -> P7 diagnostic.
//!
//! This executes one real layer-0, position-one graph from the admitted
//! streamed artifact. Static P7 controls are staged directly from verified
//! reader reads and are bound to the exact layer/token/position source contract;
//! it does not fabricate an execution-context decode position. P4B remains explicitly
//! Numeric Parity V2.1-only, so this program never promotes its child output
//! to exact storage parity, a decoder runtime, HCLI endpoint, generation, or
//! TPS result. It writes no receipt: its create-new JSON output is an
//! unsealed, source-bound diagnostic that a separate conservative receipt
//! producer may validate and seal.

#[cfg(not(target_os = "macos"))]
fn main() -> Result<(), Box<dyn std::error::Error>> {
    Err(std::io::Error::other(
        "gravity_deepseek_v4_layer0_position1_p7_device requires macOS Metal",
    )
    .into())
}

#[cfg(target_os = "macos")]
mod macos {
    use hawking_core::gravity_deepseek_v4::DeepSeekV4FullStreamReader;
    use hawking_core::gravity_deepseek_v4_expert_cache::{
        resolve_expert_bundle, DeepSeekV4ExpertBundleCache, ExpertBundleKey,
    };
    use hawking_core::gravity_deepseek_v4_layer0_attention::hc_attn_post_source_algorithm;
    use hawking_core::gravity_deepseek_v4_layer0_continuation::{POSITION1, POSITION1_TOKEN_ID};
    use hawking_core::gravity_deepseek_v4_layer0_moe::{
        layer0_mhc_ffn_control_f64_authority, layer0_mhc_ffn_post_f64_authority,
        layer0_moe_body_f32_oracle_for_token, layer0_moe_body_f64_authority_for_token,
        layer0_moe_successor_cpu_oracle, Layer0MoeCombineOrder, LAYER0_FFN_NORM_WEIGHT,
        LAYER0_HC_FFN_BASE, LAYER0_HC_FFN_FN, LAYER0_HC_FFN_SCALE,
    };
    use hawking_core::gravity_deepseek_v4_layer0_position1_ffn::verify_layer0_position1_full_ffn_source_anchors;
    use hawking_core::gravity_deepseek_v4_layer0_prefix::{
        HC_FLAT_WIDTH, HC_MIX_WIDTH, HC_MULT, HIDDEN_SIZE,
    };
    use hawking_core::gravity_deepseek_v4_p4b_device::{
        DeepSeekV4Layer0P4bDeviceExecutor, DSV4F_P4B_DEVICE_DISPATCHES,
    };
    use hawking_core::gravity_deepseek_v4_p6_device::{
        DeepSeekV4Layer0P6MetalExecutor, DSV4F_P6_DEVICE_COMMAND_BUFFERS,
        DSV4F_P6_DEVICE_COMPUTE_ENCODERS, DSV4F_P6_DEVICE_CPU_VISIBLE_WAITS,
        DSV4F_P6_DEVICE_DISPATCHES,
    };
    use hawking_core::gravity_deepseek_v4_p7_composition::{
        DeepSeekV4P7FfnSourceContract, DeepSeekV4P7SourceTensorBinding,
    };
    use hawking_core::gravity_deepseek_v4_p7_device::{
        DeepSeekV4P7BoundedDeviceExecutor, DSV4F_P7_OWNED_COMMAND_BUFFERS,
        DSV4F_P7_OWNED_COMPUTE_ENCODERS, DSV4F_P7_OWNED_CPU_VISIBLE_WAITS,
        DSV4F_P7_OWNED_DEVICE_DISPATCHES,
    };
    use hawking_core::gravity_deepseek_v4_runtime_spine::DeepSeekV4StagedTensor;
    use hawking_core::metal::{
        MetalContext, PhysicalTraceGuard, PhysicalTraceIdentity, SHADER_DEEPSEEK_V4_P7,
    };
    use hawking_core::numeric_parity::{score_pair, Bounds};
    use serde_json::{json, Value};
    use sha2::{Digest, Sha256};
    use std::error::Error;
    use std::fs::{self, File, OpenOptions};
    use std::io::{Read, Write};
    use std::path::{Path, PathBuf};
    use std::process::Command;
    use std::time::{SystemTime, UNIX_EPOCH};

    const DIAGNOSTIC_SCHEMA: &str =
        "hawking.gravity.deepseek_v4.p7_layer0_position1_device_diagnostic.v3";
    const STATUS: &str = "P7_LAYER0_POSITION1_REAL_METAL_DEVICE_GRAPH_UNSEALED_DIAGNOSTIC_V3_NUMERIC_PARITY_V2_1_ONLY_NOT_EXACT_STORAGE_NOT_RUNTIME";
    const P4B_RECEIPT_SCHEMA: &str =
        "hawking.gravity.deepseek_v4.p4b_position1_complete_attention_metal.v1";
    const P4B_RECEIPT_STATUS: &str =
        "PASS_REAL_METAL_P4B_POSITION1_COMPLETE_ATTENTION_PARITY_NOT_RUNTIME";
    const P1_FFN_RECEIPT_SCHEMA: &str =
        "hawking.gravity.deepseek_v4.layer0_position1_full_ffn_cpu_oracle.v1";
    const P1_FFN_RECEIPT_STATUS: &str =
        "PASS_SOURCE_DERIVED_CPU_LAYER0_POSITION1_FULL_FFN_NOT_INDEPENDENT_UPSTREAM_RUNTIME_PARITY";
    const P6A_RECEIPT_SCHEMA: &str =
        "hawking.gravity.deepseek_v4.layer0_moe_metal_p6a_full_route_wave.v1";
    const P6A_RECEIPT_STATUS: &str =
        "PASS_REAL_METAL_DEVICE_GATE_ROUTE_FULL_SIX_EXPERT_WAVE_NOT_FULL_RUNTIME";
    const EXPECTED_ROUTE_IDS_BY_TOP_SLOT: [u32; 6] = [72, 168, 184, 142, 174, 177];
    const EXPECTED_NUMERIC_COMBINE_ORDER: [(u32, u32); 6] =
        [(0, 72), (3, 142), (1, 168), (4, 174), (5, 177), (2, 184)];

    type ProbeResult<T> = Result<T, Box<dyn Error>>;

    struct Args {
        artifact: PathBuf,
        p4b_receipt: PathBuf,
        p1_ffn_receipt: PathBuf,
        p6a_receipt: PathBuf,
        out: PathBuf,
    }

    #[derive(Clone)]
    struct ReceiptBinding {
        path: PathBuf,
        file_sha256: String,
        seal_sha256: String,
        schema: String,
        status: String,
        transitive_p1_attention_seal_sha256: Option<String>,
    }

    #[derive(Default)]
    struct RunAccounting {
        source_control_staging_reads: u64,
        source_control_staging_bytes: u64,
        in_graph_host_handoffs: u64,
        activation_host_handoffs: u64,
        route_host_handoffs: u64,
        kv_state_host_handoffs: u64,
        fallback_paths: u64,
        post_completion_readbacks: u64,
        post_completion_readback_bytes: u64,
    }

    impl RunAccounting {
        fn note_post_completion_readback(&mut self, bytes: usize) -> ProbeResult<()> {
            self.post_completion_readbacks = self
                .post_completion_readbacks
                .checked_add(1)
                .ok_or("post-completion readback count overflow")?;
            self.post_completion_readback_bytes = self
                .post_completion_readback_bytes
                .checked_add(u64::try_from(bytes).map_err(|_| "readback byte conversion")?)
                .ok_or("post-completion readback bytes overflow")?;
            Ok(())
        }
    }

    pub fn run() -> ProbeResult<()> {
        if std::env::args_os().skip(1).any(|arg| {
            let flag = arg.to_string_lossy();
            flag == "--help" || flag == "-h"
        }) {
            println!("{}", usage());
            return Ok(());
        }
        let args = parse_args()?;
        let reader = DeepSeekV4FullStreamReader::admit(&args.artifact)?;
        verify_layer0_position1_full_ffn_source_anchors(&reader)?;
        let p4b_binding = bind_sealed_component(
            &reader,
            &args.p4b_receipt,
            "DSV4F_P4B_POSITION1_COMPLETE_ATTENTION_METAL-v1.json",
            P4B_RECEIPT_SCHEMA,
            P4B_RECEIPT_STATUS,
            true,
        )?;
        let p1_ffn_binding = bind_sealed_component(
            &reader,
            &args.p1_ffn_receipt,
            "DSV4F_LAYER0_POSITION1_FULL_FFN_CPU_ORACLE-v1.json",
            P1_FFN_RECEIPT_SCHEMA,
            P1_FFN_RECEIPT_STATUS,
            false,
        )?;
        let p6a_binding = bind_sealed_component(
            &reader,
            &args.p6a_receipt,
            "DSV4F_LAYER0_MOE_METAL_P6A-v1.json",
            P6A_RECEIPT_SCHEMA,
            P6A_RECEIPT_STATUS,
            false,
        )?;
        let executable_provenance = executable_provenance()?;
        let source_code_provenance = source_code_provenance()?;
        let host_platform = host_platform_provenance()?;
        let run_nonce = secure_run_nonce()?;
        let run_started_unix_ns = unix_time_ns()?;

        // Hash routing makes this verified `tid2eid[19923]` row static. The
        // device P6 preparation independently reads and validates the same
        // source row; same-input CPU/F64 arithmetic is computed only after
        // P7 has exposed its completed actual FFn-norm input below.
        let expected_ids = EXPECTED_ROUTE_IDS_BY_TOP_SLOT.to_vec();

        let (source, ffn_norm, mhc_ffn) = stage_verified_p7_controls(&reader)?;
        let source_control_staging_bytes = ffn_norm
            .bytes
            .len()
            .checked_add(
                mhc_ffn
                    .iter()
                    .map(|tensor| tensor.bytes.len())
                    .sum::<usize>(),
            )
            .ok_or("source-control staging bytes overflow")?;
        let mut accounting = RunAccounting {
            source_control_staging_reads: 4,
            source_control_staging_bytes: u64::try_from(source_control_staging_bytes)
                .map_err(|_| "source-control staging byte conversion")?,
            ..RunAccounting::default()
        };
        let required_hot_bytes = required_hot_cache_bytes(&reader, &expected_ids)?;
        let mut cache = DeepSeekV4ExpertBundleCache::new(required_hot_bytes, 0)?;

        let metal = MetalContext::new_with_trace(true)?;
        let mut p4b = DeepSeekV4Layer0P4bDeviceExecutor::prepare(&metal, &reader)?;
        let p6 =
            DeepSeekV4Layer0P6MetalExecutor::prepare_for_p7(&metal, &reader, &mut cache, &source)?;
        let p6_bindings = p6.source_bindings().clone();
        let cache_after_prepare = cache.state();
        if p6_bindings.selected_expert_ids_top_slot_order != EXPECTED_ROUTE_IDS_BY_TOP_SLOT {
            return Err(
                "P6 static source plan does not preserve pinned tid2eid top-slot order".into(),
            );
        }
        let numeric_order: Vec<(u32, u32)> = p6_bindings
            .resident_experts_numeric_source_order
            .iter()
            .map(|binding| (binding.source_top_slot, binding.expert_id))
            .collect();
        if numeric_order.as_slice() != EXPECTED_NUMERIC_COMBINE_ORDER {
            return Err("P6 resident experts do not preserve source numeric combine order".into());
        }
        let mut p7 = DeepSeekV4P7BoundedDeviceExecutor::prepare(
            &metal,
            source.clone(),
            &ffn_norm,
            &mhc_ffn,
            Box::new(p6),
        )?;

        // Preparation may allocate/upload and compile, but it must issue no
        // device graph command buffer. Start the graph accounting only here.
        let _ = metal.drain_trace();
        let _ = metal.drain_stats();
        let interval_id = sha256_join(&[
            &run_nonce,
            reader.manifest_seal_sha256(),
            &executable_provenance
                .get("sha256")
                .and_then(Value::as_str)
                .ok_or("executable provenance has no sha256")?,
            "dsv4f_p7_layer0_position1_bounded_device_graph_v3",
        ]);
        let physical_trace = PhysicalTraceGuard::begin(PhysicalTraceIdentity::new(
            interval_id.clone(),
            run_nonce.clone(),
            "dsv4f_p7_layer0_position1".to_owned(),
            "p4b_p7_p6_p7_bounded_graph".to_owned(),
            Some(1),
            POSITION1,
        )?)?;

        let p4b_execution = p4b.execute_position1(&metal)?;
        let output = p7.execute_from_p4b(&p4b, &metal)?;
        output.validate()?;
        if output.p4b_predecessor_parity.is_exact_storage()
            || source.host_activation_handoff_permitted
            || p6_bindings.host_activation_handoff_permitted
            || p6_bindings.host_route_weight_handoff_permitted
            || p4b_execution.host_intermediate_handoff_bytes != 0
        {
            return Err(
                "bounded P7 graph violated its no-host-handoff/V2-only predecessor contract".into(),
            );
        }

        // Every graph submission completed before this point. These reads are
        // post-completion diagnostics only, never a host bridge into P7/P6.
        let p4b_attention = p4b.p7_attention_state(&metal)?;
        let attention_hc_post = read_gpu_u16(p4b_attention.attention_hc_post_bf16, HC_FLAT_WIDTH)?;
        accounting.note_post_completion_readback(HC_FLAT_WIDTH * std::mem::size_of::<u16>())?;
        let ffn_reduced = read_gpu_u16(&output.ffn_reduced_bf16, HIDDEN_SIZE)?;
        accounting.note_post_completion_readback(HIDDEN_SIZE * std::mem::size_of::<u16>())?;
        let ffn_norm = read_gpu_u16(&output.ffn_norm_bf16, HIDDEN_SIZE)?;
        accounting.note_post_completion_readback(HIDDEN_SIZE * std::mem::size_of::<u16>())?;
        let mhc_flat_rsqrt = read_gpu_f32(&output.mhc_flat_rsqrt_f32, 1)?;
        accounting.note_post_completion_readback(std::mem::size_of::<f32>())?;
        let mhc_mixes = read_gpu_f32(&output.mhc_mixes_f32, HC_MIX_WIDTH)?;
        accounting.note_post_completion_readback(HC_MIX_WIDTH * std::mem::size_of::<f32>())?;
        let mhc_pre = read_gpu_f32(&output.mhc_pre_f32, HC_MULT)?;
        accounting.note_post_completion_readback(HC_MULT * std::mem::size_of::<f32>())?;
        let mhc_post = read_gpu_f32(&output.mhc_post_f32, HC_MULT)?;
        accounting.note_post_completion_readback(HC_MULT * std::mem::size_of::<f32>())?;
        let mhc_comb = read_gpu_f32(&output.mhc_comb_f32, HC_MULT * HC_MULT)?;
        accounting.note_post_completion_readback(HC_MULT * HC_MULT * std::mem::size_of::<f32>())?;
        let child = read_gpu_u16(&output.child_hc_state_bf16, HC_FLAT_WIDTH)?;
        accounting.note_post_completion_readback(HC_FLAT_WIDTH * std::mem::size_of::<u16>())?;
        let moe = read_gpu_u16(&output.p6.moe_output_bf16, HIDDEN_SIZE)?;
        accounting.note_post_completion_readback(HIDDEN_SIZE * std::mem::size_of::<u16>())?;
        let route_ids = read_gpu_u32(&output.p6.route_ids_u32, 6)?;
        accounting.note_post_completion_readback(6 * std::mem::size_of::<u32>())?;
        let route_weights = read_gpu_f32(&output.p6.route_weights_f32, 6)?;
        accounting.note_post_completion_readback(6 * std::mem::size_of::<f32>())?;
        let gate_logits = read_gpu_f32(&output.p6.gate_logits_f32, 256)?;
        accounting.note_post_completion_readback(256 * std::mem::size_of::<f32>())?;
        let original_scores = read_gpu_f32(&output.p6.original_scores_f32, 256)?;
        accounting.note_post_completion_readback(256 * std::mem::size_of::<f32>())?;
        let route_valid = read_gpu_u32(&output.p6.route_valid_u32, 1)?;
        accounting.note_post_completion_readback(std::mem::size_of::<u32>())?;

        // Recreate the source-F32 mHC controls from precisely the completed
        // P4B BF16 state we just read, then independently accumulate those
        // controls in FP64. `layer0_moe_successor_cpu_oracle` presently
        // carries its scalar source oracle through the later MoE branch too;
        // this diagnostic reads only `ffn_hc_pre` and deliberately does not
        // inspect, compare, or claim its expert or child-post outputs.
        let same_input_mhc =
            layer0_moe_successor_cpu_oracle(&reader, POSITION1_TOKEN_ID, &attention_hc_post)?
                .ffn_hc_pre;
        let same_input_mhc_f64 = layer0_mhc_ffn_control_f64_authority(&reader, &attention_hc_post)?;
        let mhc_control_v21_bounds = p7_mhc_control_v21_bounds();
        let mhc_storage_v21_bounds = p7_bf16_storage_v21_bounds();
        let mhc_flat_rsqrt_f64 = [same_input_mhc_f64.flat_rsqrt_f64];
        let mhc_flat_rsqrt_v21 = score_pair(
            &[same_input_mhc.flat_rsqrt],
            &mhc_flat_rsqrt,
            &mhc_flat_rsqrt_f64,
            &mhc_control_v21_bounds,
        );
        let mhc_mixes_v21 = score_pair(
            &same_input_mhc.mixes_f32,
            &mhc_mixes,
            &same_input_mhc_f64.mixes_f64,
            &mhc_control_v21_bounds,
        );
        let mhc_pre_v21 = score_pair(
            &same_input_mhc.pre_f32,
            &mhc_pre,
            &same_input_mhc_f64.pre_f64,
            &mhc_control_v21_bounds,
        );
        let mhc_post_v21 = score_pair(
            &same_input_mhc.post_f32,
            &mhc_post,
            &same_input_mhc_f64.post_f64,
            &mhc_control_v21_bounds,
        );
        let mhc_comb_v21 = score_pair(
            &same_input_mhc.comb_f32,
            &mhc_comb,
            &same_input_mhc_f64.comb_f64,
            &mhc_control_v21_bounds,
        );
        let mhc_reduced_v21 = score_pair(
            &bf16_bits_f32(&same_input_mhc.reduced_bf16_bits),
            &bf16_bits_f32(&ffn_reduced),
            &bf16_bits_f64(&same_input_mhc_f64.reduced_bf16_bits),
            &mhc_storage_v21_bounds,
        );
        let same_input_mhc_v21_pass = mhc_flat_rsqrt_v21.pass
            && mhc_mixes_v21.pass
            && mhc_pre_v21.pass
            && mhc_post_v21.pass
            && mhc_comb_v21.pass
            && mhc_reduced_v21.pass;

        // Both MoE-body authorities consume precisely the BF16 row P6 used
        // on device. They stop at the source MoE BF16 store: no host value
        // from either authority can enter this device graph.
        let same_input_moe_cpu =
            layer0_moe_body_f32_oracle_for_token(&reader, POSITION1_TOKEN_ID, &ffn_norm)?;
        let same_input_moe_f64 =
            layer0_moe_body_f64_authority_for_token(&reader, POSITION1_TOKEN_ID, &ffn_norm)?;
        let same_input_route = &same_input_moe_cpu.route;
        let same_input_f64 = &same_input_moe_f64.route;
        let same_input_ids: Vec<u32> = same_input_route
            .selected_expert_ids
            .iter()
            .copied()
            .map(|id| {
                u32::try_from(id)
                    .map_err(|_| std::io::Error::other("same-input CPU route ID exceeds u32"))
            })
            .collect::<Result<_, _>>()?;
        if same_input_ids != expected_ids {
            return Err("same-input CPU route differs from pinned tid2eid top-slot row".into());
        }
        if same_input_f64.selected_expert_ids != same_input_route.selected_expert_ids {
            return Err(
                "independent F64 route differs from same-input source CPU tid2eid row".into(),
            );
        }
        let source_f32_combine_order =
            source_combine_order_u32(&same_input_moe_cpu.routed_combine_order)?;
        let fp64_combine_order =
            source_combine_order_u32(&same_input_moe_f64.routed_combine_order)?;
        if source_f32_combine_order != EXPECTED_NUMERIC_COMBINE_ORDER
            || fp64_combine_order != EXPECTED_NUMERIC_COMBINE_ORDER
            || source_f32_combine_order != fp64_combine_order
        {
            return Err(
                "same-input MoE authorities do not preserve pinned numeric expert combine order"
                    .into(),
            );
        }
        let route_v21_bounds = Bounds {
            max_meaningful_rel: 1.0e-4,
            ..Bounds::continuous_only()
        };
        let gate_logits_v21 = score_pair(
            &same_input_route.logits_f32,
            &gate_logits,
            &same_input_f64.logits_f64,
            &route_v21_bounds,
        );
        let original_scores_v21 = score_pair(
            &same_input_route.original_scores_f32,
            &original_scores,
            &same_input_f64.original_scores_f64,
            &route_v21_bounds,
        );
        let route_weights_v21 = score_pair(
            &same_input_route.selected_weights_f32,
            &route_weights,
            &same_input_f64.selected_weights_f64,
            &route_v21_bounds,
        );
        let same_input_route_v21_pass = gate_logits_v21.pass
            && original_scores_v21.pass
            && route_weights_v21.pass
            && same_input_route.logits_f32 == gate_logits
            && same_input_ids == route_ids
            && route_valid == vec![1];

        // The P6 output is a declared BF16 store. Both source-F32 and device
        // candidates therefore compare their stored BF16 rows to the
        // independently accumulated FP64 serial source-order combine.
        let moe_body_v21 = score_pair(
            &bf16_bits_f32(&same_input_moe_cpu.moe_output_bf16_bits),
            &bf16_bits_f32(&moe),
            &same_input_moe_f64.combined_f64,
            &mhc_storage_v21_bounds,
        );
        let same_input_moe_v21_pass = moe_body_v21.pass;

        // The final child comparison consumes the exact device-produced MoE
        // BF16 row and exact completed P4B attention BF16 state on both F32
        // and FP64 authority paths. Source-F32 post controls are regenerated
        // from that captured attention state; no source successor child value
        // is reused.
        let same_input_child_cpu = hc_attn_post_source_algorithm(
            &moe,
            &attention_hc_post,
            &same_input_mhc.post_f32,
            &same_input_mhc.comb_f32,
        )?;
        let same_input_child_f64 =
            layer0_mhc_ffn_post_f64_authority(&reader, &attention_hc_post, &moe)?;
        let child_v21 = score_pair(
            &bf16_bits_f32(&same_input_child_cpu),
            &bf16_bits_f32(&child),
            &same_input_child_f64.child_state_f64,
            &mhc_storage_v21_bounds,
        );
        let same_input_child_v21_pass = child_v21.pass;

        let physical_counts = physical_trace.counts();
        drop(physical_trace);
        let run_finished_unix_ns = unix_time_ns()?;
        let trace = metal.drain_trace();
        let (graph_buffers_created, graph_bytes_allocated, committed_command_buffers) =
            metal.drain_stats();
        let expected_command_buffers = p4b_execution.actual_command_buffers
            + DSV4F_P7_OWNED_COMMAND_BUFFERS
            + DSV4F_P6_DEVICE_COMMAND_BUFFERS;
        let expected_waits = p4b_execution.actual_cpu_visible_waits
            + DSV4F_P7_OWNED_CPU_VISIBLE_WAITS
            + DSV4F_P6_DEVICE_CPU_VISIBLE_WAITS;
        let expected_dispatches = p4b_execution.actual_gpu_dispatches
            + DSV4F_P7_OWNED_DEVICE_DISPATCHES
            + DSV4F_P6_DEVICE_DISPATCHES;
        let expected_encoders = p4b_execution.actual_compute_encoders
            + DSV4F_P7_OWNED_COMPUTE_ENCODERS
            + DSV4F_P6_DEVICE_COMPUTE_ENCODERS;
        if p4b_execution.actual_gpu_dispatches != DSV4F_P4B_DEVICE_DISPATCHES
            || expected_command_buffers != 5
            || expected_waits != 5
            || expected_dispatches != 96
            || expected_encoders != 46
            || committed_command_buffers != expected_command_buffers
            || trace.len() != expected_command_buffers
            || physical_counts.command_count as usize != expected_command_buffers
            || physical_counts.encoder_count as usize != expected_encoders
        {
            return Err(format!(
                "bounded P7 topology mismatch: p4b={:?}, expected cb/waits/dispatches/encoders={expected_command_buffers}/{expected_waits}/{expected_dispatches}/{expected_encoders}, observed commits={}, trace batches={}, physical cb/encoders={}/{}",
                p4b_execution,
                committed_command_buffers,
                trace.len(),
                physical_counts.command_count,
                physical_counts.encoder_count,
            )
            .into());
        }
        if route_valid != vec![1] {
            return Err(
                format!("P6 route kernel marked its output invalid: {route_valid:?}").into(),
            );
        }
        if accounting.in_graph_host_handoffs != 0
            || accounting.activation_host_handoffs != 0
            || accounting.route_host_handoffs != 0
            || accounting.kv_state_host_handoffs != 0
            || accounting.fallback_paths != 0
        {
            return Err(
                "bounded P7 run accounting observed an in-graph host handoff or fallback".into(),
            );
        }

        let unsigned = json!({
            "schema": DIAGNOSTIC_SCHEMA,
            "status": STATUS,
            "unsealed": true,
            "artifact": {
                "path": reader.artifact_root().display().to_string(),
                "manifest_seal_sha256": reader.manifest_seal_sha256(),
                "manifest_file_sha256": reader.manifest_file_sha256(),
                "restart_receipt_seal_sha256": reader.restart_seal_sha256(),
                "source_repository": reader.source_identity().repository,
                "source_revision": reader.source_identity().revision,
                "source_parent_retained": false,
            },
            "source_code_provenance": source_code_provenance,
            "component_receipt_bindings": {
                "all_are_component_or_topology_bindings_not_direct_numeric_ancestry": true,
                "p4b_bounded_attention_predecessor": {
                    "receipt": receipt_binding_json(&p4b_binding),
                    "relation": "bounded all-device input-state/component provenance only; the actual P4B input is classified separately as Numeric Parity V2.1-only and is never promoted to exact-storage ancestry",
                    "transitive_position1_complete_attention_cpu_oracle_seal_sha256": p4b_binding.transitive_p1_attention_seal_sha256.clone(),
                },
                "p1_full_ffn_cpu_oracle": {
                    "receipt": receipt_binding_json(&p1_ffn_binding),
                    "relation": "source-semantic component anchor only; it is not the direct numerical parent of this post-completion actual-input P7 graph",
                },
                "p6a_full_route_wave": {
                    "receipt": receipt_binding_json(&p6a_binding),
                    "relation": "device component/topology reference only; its deterministic fixture is not this graph's actual FFn-norm input and it is not direct numerical ancestry",
                },
            },
            "run_provenance": {
                "run_nonce_sha256": run_nonce,
                "run_started_unix_ns_text": run_started_unix_ns,
                "run_finished_unix_ns_text": run_finished_unix_ns,
                "process_id": std::process::id(),
                "executable": executable_provenance,
                "host_platform": host_platform,
                "physical_trace": {
                    "interval_id": interval_id,
                    "phase": "dsv4f_p7_layer0_position1",
                    "role": "p4b_p7_p6_p7_bounded_graph",
                    "batch": 1,
                    "iteration": POSITION1,
                    "command_buffers": physical_counts.command_count,
                    "compute_encoders": physical_counts.encoder_count,
                },
            },
            "run_accounting": {
                "source_control_staging_reads_before_graph": accounting.source_control_staging_reads,
                "source_control_staging_bytes_before_graph": accounting.source_control_staging_bytes,
                "in_graph_host_handoffs": accounting.in_graph_host_handoffs,
                "activation_host_handoffs": accounting.activation_host_handoffs,
                "route_host_handoffs": accounting.route_host_handoffs,
                "kv_state_host_handoffs": accounting.kv_state_host_handoffs,
                "fallback_paths": accounting.fallback_paths,
                "post_completion_diagnostic_readbacks": accounting.post_completion_readbacks,
                "post_completion_diagnostic_readback_bytes": accounting.post_completion_readback_bytes,
                "post_completion_readbacks_are_not_graph_handoffs": true,
                "fallback_policy": "no alternate host/CPU execution path is selected by this bounded runner; any executor error aborts before publication",
            },
            "scope": {
                "artifact_manifest_seal_sha256": reader.manifest_seal_sha256(),
                "source_revision": reader.source_identity().revision,
                "layer": output.layer,
                "token_id": output.token_id,
                "token_position": output.token_position,
                "device": metal.device_name(),
                "p1_trace_provenance": "verified layer-0/position-1 source anchors plus direct verified-reader P7 static-control staging; same-input CPU route is recomputed only after post-completion FFn-norm diagnostic readback",
                "source_parent_retained": false,
            },
            "p4b_predecessor": {
                "classification": output.p4b_predecessor_parity.as_str(),
                "exact_storage": false,
                "policy": "The single reusable P4B label remains Numeric Parity V2.1 only; this graph cannot be presented as exact-storage P7 evidence.",
            },
            "source_controls": {
                "ffn_norm": source_binding_json(&source.ffn_norm),
                "hc_ffn_fn": source_binding_json(&source.hc_ffn_fn),
                "hc_ffn_base": source_binding_json(&source.hc_ffn_base),
                "hc_ffn_scale": source_binding_json(&source.hc_ffn_scale),
                "staging": "direct verified reader controls; no fabricated PreparedDecodeInput position",
            },
            "p6_residency": {
                "hot_capacity_bytes": cache_after_prepare.hot_capacity_bytes,
                "hot_resident_bytes": cache_after_prepare.hot_resident_bytes,
                "cold_resident_bytes": cache_after_prepare.cold_resident_bytes,
                "hot_expert_keys": cache_after_prepare.hot_keys_lru_to_mru.iter().map(|key| json!({"layer": key.layer, "expert": key.expert})).collect::<Vec<_>>(),
                "source_bundle_loads": cache_after_prepare.counters.source_bundle_loads,
                "source_payload_bytes_returned": cache_after_prepare.counters.source_payload_bytes_returned,
                "top_slot_route_ids": p6_bindings.selected_expert_ids_top_slot_order,
                "numeric_source_combine_order": numeric_order,
            },
            "actual_graph_topology": {
                "command_buffers": committed_command_buffers,
                "cpu_visible_completion_waits": expected_waits,
                "gpu_dispatches": expected_dispatches,
                "compute_encoders": expected_encoders,
                "p4b": {
                    "command_buffers": p4b_execution.actual_command_buffers,
                    "cpu_visible_completion_waits": p4b_execution.actual_cpu_visible_waits,
                    "gpu_dispatches": p4b_execution.actual_gpu_dispatches,
                    "compute_encoders": p4b_execution.actual_compute_encoders,
                },
                "p7_owned": {
                    "command_buffers": DSV4F_P7_OWNED_COMMAND_BUFFERS,
                    "cpu_visible_completion_waits": DSV4F_P7_OWNED_CPU_VISIBLE_WAITS,
                    "gpu_dispatches": DSV4F_P7_OWNED_DEVICE_DISPATCHES,
                    "compute_encoders": DSV4F_P7_OWNED_COMPUTE_ENCODERS,
                },
                "p6": {
                    "command_buffers": DSV4F_P6_DEVICE_COMMAND_BUFFERS,
                    "cpu_visible_completion_waits": DSV4F_P6_DEVICE_CPU_VISIBLE_WAITS,
                    "gpu_dispatches": DSV4F_P6_DEVICE_DISPATCHES,
                    "compute_encoders": DSV4F_P6_DEVICE_COMPUTE_ENCODERS,
                },
                "buffers_created_during_graph": graph_buffers_created,
                "bytes_allocated_during_graph": graph_bytes_allocated,
                "physical_trace_command_buffers": physical_counts.command_count,
                "physical_trace_compute_encoders": physical_counts.encoder_count,
                "trace_samples": trace.len(),
                "ordered_command_batches": ordered_command_batches(&trace),
                "optimization_status": "five command buffers/five waits: within the <=8/token intermediate target only; not <=3, replayable, persistent, runtime, or TPS evidence",
            },
            "post_completion_device_diagnostics": {
                "route_valid_u32": route_valid,
                "actual_p4b_attention_hc_post_bf16": observed_bf16(&attention_hc_post),
                "same_actual_input_mhc_controls": {
                    "scope": "P7 hc_ffn_pre, Sinkhorn-control, and reduced-BF16-store diagnostics from the exact post-completion P4B BF16[4,4096] input. This object excludes the later P6 body and P7 mHC-post child comparison, which are reported separately below.",
                    "reference_authority": "independent FP64 mHC linear/Sinkhorn accumulation from verified F32 controls and the exact device-produced P4B BF16 attention state; reduced storage is its declared FP64-to-source-F32-to-BF16 projection.",
                    "same_input_source_f32": "the source-F32 successor is recomputed from the same captured attention state; its non-mHC values are dropped and not used for this comparison.",
                    "source_f32_vs_device_storage": {
                        "ffn_reduced_bf16": diagnostic_u16(&same_input_mhc.reduced_bf16_bits, &ffn_reduced),
                    },
                    "fp64_projected_storage": {
                        "ffn_reduced_bf16": diagnostic_u16(&same_input_mhc_f64.reduced_bf16_bits, &ffn_reduced),
                    },
                    "numeric_parity_v2_1": {
                        "schema": "hawking.numeric_parity.v2_1",
                        "continuous_control_bounds": mhc_control_v21_bounds,
                        "bf16_storage_bounds": mhc_storage_v21_bounds,
                        "flat_rsqrt_f32": mhc_flat_rsqrt_v21,
                        "mixes_f32": mhc_mixes_v21,
                        "pre_f32": mhc_pre_v21,
                        "post_control_f32": mhc_post_v21,
                        "comb_f32": mhc_comb_v21,
                        "reduced_bf16_projected_store": mhc_reduced_v21,
                        "all_scored_mhc_controls_pass": same_input_mhc_v21_pass,
                    },
                },
                "actual_ffn_norm_bf16": observed_bf16(&ffn_norm),
                "same_actual_input_cpu_route": {
                    "route_ids": diagnostic_u32(&same_input_ids, &route_ids),
                    "route_weights": diagnostic_f32(&same_input_route.selected_weights_f32, &route_weights),
                    "gate_logits": diagnostic_f32(&same_input_route.logits_f32, &gate_logits),
                    "original_scores": diagnostic_f32(&same_input_route.original_scores_f32, &original_scores),
                },
                "same_actual_input_numeric_parity_v2_1": {
                    "scope": "P6 Gate/hash-route controls only, from the exact post-completion P7 FFn-norm BF16 input; the full P6 body and P7 child checks are separately scoped below.",
                    "reference_authority": "independent FP64 Gate/sqrt-softplus/normalization from verified raw BF16 Gate weights and the same device-produced BF16 input",
                    "bounds": route_v21_bounds,
                    "gate_logits": gate_logits_v21,
                    "original_scores": original_scores_v21,
                    "selected_weights": route_weights_v21,
                    "exact_gate_logit_bits": same_input_route.logits_f32 == gate_logits,
                    "exact_tid2eid_ids": same_input_ids == route_ids,
                    "route_valid_word": route_valid,
                    "continuous_and_discrete_controls_pass": same_input_route_v21_pass,
                },
                "same_actual_input_moe_body": {
                    "scope": "P6 Gate/hash route, six selected native-FP4 routed experts, permanent native-FP8 shared expert, source numeric-order combine, and the MoE BF16 store. Both authorities consume the exact post-completion P7 FFn-norm BF16 row; neither evaluates P7 mHC-post or a decoder runtime.",
                    "reference_authority": "independent FP64 native FP4/FP8 decode, BF16 source-store boundaries, SwiGLU, and serial source-order combine from verified windows.",
                    "source_numeric_combine_order": {
                        "device_p6": numeric_order,
                        "source_f32": source_f32_combine_order,
                        "fp64_authority": fp64_combine_order,
                        "all_match": true,
                    },
                    "source_f32_vs_device_storage": {
                        "moe_output_bf16": diagnostic_u16(&same_input_moe_cpu.moe_output_bf16_bits, &moe),
                    },
                    "fp64_projected_storage": {
                        "moe_output_bf16": diagnostic_u16(&same_input_moe_f64.moe_output_bf16_bits, &moe),
                    },
                    "numeric_parity_v2_1": {
                        "schema": "hawking.numeric_parity.v2_1",
                        "bf16_storage_bounds": mhc_storage_v21_bounds,
                        "moe_output_bf16_store": moe_body_v21,
                        "source_f32_and_device_score_pass": same_input_moe_v21_pass,
                    },
                },
                "same_actual_input_mhc_post_child": {
                    "scope": "P7 mHC-post from the exact completed P4B attention BF16 state and exact completed P6 MoE BF16 row through the child BF16[4,4096] store. This remains a bounded layer-0/token-19923 diagnostic, not a causal continuation or runtime proof.",
                    "reference_authority": "independent FP64 source-order mHC-post using independently accumulated controls and the exact captured BF16 attention/MoE inputs.",
                    "source_f32_vs_device_storage": {
                        "child_hc_state_bf16": diagnostic_u16(&same_input_child_cpu, &child),
                    },
                    "fp64_projected_storage": {
                        "child_hc_state_bf16": diagnostic_u16(&same_input_child_f64.child_state_bf16_bits, &child),
                    },
                    "numeric_parity_v2_1": {
                        "schema": "hawking.numeric_parity.v2_1",
                        "bf16_storage_bounds": mhc_storage_v21_bounds,
                        "child_hc_state_bf16_store": child_v21,
                        "source_f32_and_device_score_pass": same_input_child_v21_pass,
                    },
                },
                "moe_output_bf16": observed_bf16(&moe),
                "child_hc_state_bf16": observed_bf16(&child),
                "note": "Every scored section has explicit same-input scope: mHC-pre controls consume the captured P4B BF16 state; P6 controls/body consume the captured P7 FFn-norm BF16 row; mHC-post consumes the captured P4B attention and P6 MoE BF16 rows. P4B nevertheless remains Numeric Parity V2.1-only, so this unsealed diagnostic cannot be relabelled exact-storage P7 or runtime evidence.",
            },
            "claim_boundary": "Real Metal dispatches occurred in one fresh, physically attributed bounded layer-0/position-1 all-device graph with no activation/route/KV/child handoff during execution and no fallback path taken. Post-completion readback supports the explicitly scoped same-input mHC-pre, P6-route/body, and mHC-post child diagnostics only. This unsealed diagnostic establishes Numeric Parity V2.1-only evidence conditional on a P4B Numeric Parity V2.1-only input; it does not establish P7 exact storage parity, a complete decoder layer, a registered 43-layer causal runtime, first token, continuation, HCLI endpoint, or BASE_TRUE_TPS."
        });
        let mut response = decimal_strings(unsigned);
        let canonical_unsigned_sha256 = sha256(&canonical_json(&response));
        response
            .as_object_mut()
            .ok_or("P7 v3 diagnostic root is not an object")?
            .insert(
                "canonical_unsigned_sha256".to_owned(),
                Value::String(canonical_unsigned_sha256),
            );
        let rendered = serde_json::to_string_pretty(&response)?;
        write_new_unsealed_diagnostic(&args.out, &rendered)?;
        println!("{rendered}");
        Ok(())
    }

    fn parse_args() -> ProbeResult<Args> {
        let mut artifact = None;
        let mut p4b_receipt = None;
        let mut p1_ffn_receipt = None;
        let mut p6a_receipt = None;
        let mut out = None;
        let mut args = std::env::args_os().skip(1);
        while let Some(flag) = args.next() {
            match flag.to_string_lossy().as_ref() {
                "--artifact" => artifact = args.next().map(PathBuf::from),
                "--p4b-receipt" => p4b_receipt = args.next().map(PathBuf::from),
                "--p1-ffn-receipt" => p1_ffn_receipt = args.next().map(PathBuf::from),
                "--p6a-receipt" => p6a_receipt = args.next().map(PathBuf::from),
                "--out" => out = args.next().map(PathBuf::from),
                other => return Err(format!("unknown argument {other}").into()),
            }
        }
        let artifact = artifact.ok_or("--artifact required")?;
        let p4b_receipt = p4b_receipt.ok_or("--p4b-receipt required")?;
        let p1_ffn_receipt = p1_ffn_receipt.ok_or("--p1-ffn-receipt required")?;
        let p6a_receipt = p6a_receipt.ok_or("--p6a-receipt required")?;
        let out = out.ok_or("--out required")?;
        if !artifact.is_absolute()
            || !p4b_receipt.is_absolute()
            || !p1_ffn_receipt.is_absolute()
            || !p6a_receipt.is_absolute()
            || !out.is_absolute()
        {
            return Err("--artifact, receipt inputs, and --out must be absolute paths".into());
        }
        Ok(Args {
            artifact,
            p4b_receipt,
            p1_ffn_receipt,
            p6a_receipt,
            out,
        })
    }

    fn usage() -> &'static str {
        "usage: gravity_deepseek_v4_layer0_position1_p7_device \\\n+  --artifact <absolute full Gravity dir> \\\n+  --p4b-receipt <absolute DSV4F_P4B_POSITION1_COMPLETE_ATTENTION_METAL-v1.json> \\\n+  --p1-ffn-receipt <absolute DSV4F_LAYER0_POSITION1_FULL_FFN_CPU_ORACLE-v1.json> \\\n+  --p6a-receipt <absolute DSV4F_LAYER0_MOE_METAL_P6A-v1.json> \\\n+  --out <absolute new unsealed P7 v3 diagnostic.json>"
    }

    fn bind_sealed_component(
        reader: &DeepSeekV4FullStreamReader,
        input: &Path,
        expected_basename: &str,
        expected_schema: &str,
        expected_status: &str,
        require_transitive_p1_attention: bool,
    ) -> ProbeResult<ReceiptBinding> {
        if input.file_name().and_then(|name| name.to_str()) != Some(expected_basename) {
            return Err(
                format!("wrong component receipt basename; expected {expected_basename}").into(),
            );
        }
        if fs::symlink_metadata(input)?.file_type().is_symlink() {
            return Err("component receipt symlinks are not admitted".into());
        }
        let path = fs::canonicalize(input)?;
        if !fs::metadata(&path)?.is_file() {
            return Err("component receipt is not a regular file".into());
        }
        let raw = fs::read(&path)?;
        let value: Value = serde_json::from_slice(&raw)?;
        verify_sealed_json(&value, expected_basename)?;
        let schema = text_at(&value, &["schema"])?;
        let status = text_at(&value, &["status"])?;
        if schema != expected_schema || status != expected_status {
            return Err(format!(
                "component receipt schema/status mismatch: observed={schema}/{status}"
            )
            .into());
        }
        if text_at(&value, &["artifact", "manifest_seal_sha256"])? != reader.manifest_seal_sha256()
            || text_at(&value, &["artifact", "manifest_file_sha256"])?
                != reader.manifest_file_sha256()
        {
            return Err(
                "component receipt artifact binding differs from the admitted stream".into(),
            );
        }
        let transitive_p1_attention_seal_sha256 = if require_transitive_p1_attention {
            let predecessor = value_at(&value, &["predecessors", "position1_complete_cpu_oracle"])?;
            let predecessor_path = text_at(predecessor, &["path"])?;
            if Path::new(predecessor_path)
                .file_name()
                .and_then(|name| name.to_str())
                != Some("DSV4F_LAYER0_POSITION1_COMPLETE_ATTENTION_CPU_ORACLE-v1.json")
            {
                return Err(
                    "P4B predecessor does not bind the expected P1 attention CPU receipt".into(),
                );
            }
            let predecessor_file_sha256 = text_at(predecessor, &["file_sha256"])?;
            let predecessor_seal_sha256 = text_at(predecessor, &["seal_sha256"])?;
            if !is_sha256(predecessor_file_sha256) || !is_sha256(predecessor_seal_sha256) {
                return Err("P4B transitive P1 attention receipt hash is malformed".into());
            }
            Some(predecessor_seal_sha256.to_owned())
        } else {
            None
        };
        Ok(ReceiptBinding {
            path,
            file_sha256: sha256(&raw),
            seal_sha256: text_at(&value, &["seal_sha256"])?.to_owned(),
            schema: schema.to_owned(),
            status: status.to_owned(),
            transitive_p1_attention_seal_sha256,
        })
    }

    fn receipt_binding_json(binding: &ReceiptBinding) -> Value {
        json!({
            "path": binding.path.display().to_string(),
            "file_sha256": binding.file_sha256,
            "seal_sha256": binding.seal_sha256,
            "schema": binding.schema,
            "status": binding.status,
        })
    }

    fn value_at<'a>(value: &'a Value, path: &[&str]) -> ProbeResult<&'a Value> {
        let mut current = value;
        for key in path {
            current = current
                .get(*key)
                .ok_or_else(|| format!("receipt is missing {}", path.join(".")))?;
        }
        Ok(current)
    }

    fn text_at<'a>(value: &'a Value, path: &[&str]) -> ProbeResult<&'a str> {
        value_at(value, path)?
            .as_str()
            .ok_or_else(|| format!("receipt {} is not text", path.join(".")).into())
    }

    fn is_sha256(value: &str) -> bool {
        value.len() == 64
            && value
                .bytes()
                .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte))
    }

    fn is_git_revision(value: &str) -> bool {
        matches!(value.len(), 40 | 64)
            && value
                .bytes()
                .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte))
    }

    fn verify_sealed_json(value: &Value, label: &str) -> ProbeResult<()> {
        let recorded = text_at(value, &["seal_sha256"])?;
        if !is_sha256(recorded) {
            return Err(format!("{label} has malformed seal_sha256").into());
        }
        let mut unsigned = value.clone();
        unsigned
            .as_object_mut()
            .ok_or_else(|| format!("{label} root is not an object"))?
            .remove("seal_sha256")
            .ok_or_else(|| format!("{label} lacks seal_sha256"))?;
        if sha256(&canonical_json(&unsigned)) != recorded {
            return Err(format!("{label} canonical seal mismatch").into());
        }
        Ok(())
    }

    fn executable_provenance() -> ProbeResult<Value> {
        let path = std::env::current_exe()?;
        if fs::symlink_metadata(&path)?.file_type().is_symlink() {
            return Err("current executable symlink is not admitted for run provenance".into());
        }
        let path = fs::canonicalize(path)?;
        let metadata = fs::metadata(&path)?;
        if !metadata.is_file() || metadata.len() == 0 {
            return Err("current executable is not a nonempty regular file".into());
        }
        let build_profile = if path
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
            return Err("cannot derive build profile from current executable path".into());
        };
        Ok(json!({
            "path": path.display().to_string(),
            "sha256": sha256(&fs::read(&path)?),
            "bytes": metadata.len(),
            "build_profile": build_profile,
            "cargo_package": env!("CARGO_PKG_NAME"),
            "cargo_package_version": env!("CARGO_PKG_VERSION"),
        }))
    }

    fn source_code_provenance() -> ProbeResult<Value> {
        let root = Path::new(env!("CARGO_MANIFEST_DIR"));
        let relative_files = [
            "examples/gravity_deepseek_v4_layer0_position1_p7_device.rs",
            "src/gravity_deepseek_v4_p7_device.rs",
            "src/gravity_deepseek_v4_p7_composition.rs",
            "src/gravity_deepseek_v4_p6_device.rs",
            "src/gravity_deepseek_v4_p4b_device.rs",
            "src/metal/mod.rs",
            "shaders/deepseek_v4_p7.metal",
        ];
        let mut files = serde_json::Map::new();
        for relative in relative_files {
            let path = root.join(relative);
            files.insert(relative.to_owned(), Value::String(sha256(&fs::read(path)?)));
        }
        let shader_source_sha256 = files
            .get("shaders/deepseek_v4_p7.metal")
            .and_then(Value::as_str)
            .ok_or("P7 shader source hash absent")?
            .to_owned();
        let shader_embedded_sha256 = sha256(SHADER_DEEPSEEK_V4_P7.as_bytes());
        if shader_source_sha256 != shader_embedded_sha256 {
            return Err("embedded P7 shader differs from current shader source file".into());
        }
        let checkout_revision = command_stdout("git", &["rev-parse", "HEAD"], false)?;
        if !is_git_revision(&checkout_revision) {
            return Err("git checkout revision is not a 40- or 64-hex commit ID".into());
        }
        let worktree_porcelain = command_stdout("git", &["status", "--porcelain=v1"], true)?;
        Ok(json!({
            "checkout_revision": checkout_revision,
            "worktree_porcelain_sha256": sha256(worktree_porcelain.as_bytes()),
            "source_files_sha256": files,
            "p7_shader_embedded_sha256": shader_embedded_sha256,
            "p7_shader_embedded_matches_current_source_file": true,
        }))
    }

    fn host_platform_provenance() -> ProbeResult<Value> {
        Ok(json!({
            "operating_system": std::env::consts::OS,
            "architecture": command_stdout("uname", &["-m"], false)?,
            "kernel_release": command_stdout("uname", &["-r"], false)?,
            "macos_product_version": command_stdout("sw_vers", &["-productVersion"], false)?,
            "macos_build_version": command_stdout("sw_vers", &["-buildVersion"], false)?,
        }))
    }

    fn command_stdout(program: &str, args: &[&str], allow_empty: bool) -> ProbeResult<String> {
        let output = Command::new(program).args(args).output()?;
        if !output.status.success() {
            return Err(format!("{program} {:?} failed with {}", args, output.status).into());
        }
        let text = String::from_utf8(output.stdout)?.trim().to_owned();
        if !allow_empty && text.is_empty() {
            return Err(format!("{program} {:?} produced empty stdout", args).into());
        }
        Ok(text)
    }

    fn secure_run_nonce() -> ProbeResult<String> {
        let mut entropy = [0u8; 32];
        File::open("/dev/urandom")?.read_exact(&mut entropy)?;
        Ok(sha256(&entropy))
    }

    fn unix_time_ns() -> ProbeResult<String> {
        Ok(SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .map_err(|error| format!("system clock before Unix epoch: {error}"))?
            .as_nanos()
            .to_string())
    }

    fn ordered_command_batches(trace: &[hawking_core::metal::DispatchSample]) -> Vec<Value> {
        const STAGES: [&str; 5] = [
            "P4B position-1 complete attention",
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
                    "layer_hint": sample.layer_hint,
                })
            })
            .collect()
    }

    fn write_new_unsealed_diagnostic(path: &std::path::Path, rendered: &str) -> ProbeResult<()> {
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
        let temporary = parent.join(format!(".{name}.{}.p7-v3.tmp", std::process::id()));
        if temporary.exists() {
            return Err(format!(
                "unsealed diagnostic temporary already exists {}",
                temporary.display()
            )
            .into());
        }
        let mut file = OpenOptions::new()
            .write(true)
            .create_new(true)
            .open(&temporary)
            .map_err(|error| {
                std::io::Error::new(
                    error.kind(),
                    format!(
                        "cannot create unsealed diagnostic temporary {}: {error}",
                        temporary.display()
                    ),
                )
            })?;
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

    fn canonical_json(value: &Value) -> Vec<u8> {
        let mut output = Vec::new();
        write_canonical_json(&mut output, value);
        output
    }

    fn write_canonical_json(output: &mut Vec<u8>, value: &Value) {
        match value {
            Value::Null => output.extend_from_slice(b"null"),
            Value::Bool(true) => output.extend_from_slice(b"true"),
            Value::Bool(false) => output.extend_from_slice(b"false"),
            Value::Number(number) => output.extend_from_slice(number.to_string().as_bytes()),
            Value::String(string) => output.extend_from_slice(
                serde_json::to_string(string)
                    .expect("JSON string serialization is infallible")
                    .as_bytes(),
            ),
            Value::Array(values) => {
                output.push(b'[');
                for (index, value) in values.iter().enumerate() {
                    if index != 0 {
                        output.push(b',');
                    }
                    write_canonical_json(output, value);
                }
                output.push(b']');
            }
            Value::Object(values) => {
                let mut keys = values.keys().collect::<Vec<_>>();
                keys.sort();
                output.push(b'{');
                for (index, key) in keys.into_iter().enumerate() {
                    if index != 0 {
                        output.push(b',');
                    }
                    output.extend_from_slice(
                        serde_json::to_string(key)
                            .expect("JSON key serialization is infallible")
                            .as_bytes(),
                    );
                    output.push(b':');
                    write_canonical_json(output, &values[key]);
                }
                output.push(b'}');
            }
        }
    }

    /// Preserve exact integer counters as JSON numbers while making every
    /// non-integer numerical diagnostic a text value before canonical hashing.
    /// This matches sealed-receipt conventions and avoids cross-language
    /// floating-point rendering ambiguity without suppressing unavailable
    /// timings, which remain JSON null.
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

    fn stage_verified_p7_controls(
        reader: &DeepSeekV4FullStreamReader,
    ) -> ProbeResult<(
        DeepSeekV4P7FfnSourceContract,
        DeepSeekV4StagedTensor,
        [DeepSeekV4StagedTensor; 3],
    )> {
        let ffn_norm = stage_verified_full(reader, LAYER0_FFN_NORM_WEIGHT)?;
        let hc_fn = stage_verified_full(reader, LAYER0_HC_FFN_FN)?;
        let hc_base = stage_verified_full(reader, LAYER0_HC_FFN_BASE)?;
        let hc_scale = stage_verified_full(reader, LAYER0_HC_FFN_SCALE)?;
        let source = DeepSeekV4P7FfnSourceContract {
            layer: 0,
            token_id: POSITION1_TOKEN_ID as u32,
            token_position: POSITION1,
            ffn_norm: source_binding(&ffn_norm),
            hc_ffn_fn: source_binding(&hc_fn),
            hc_ffn_base: source_binding(&hc_base),
            hc_ffn_scale: source_binding(&hc_scale),
            source_parent_retained: false,
            source_upload_required_before_execution: true,
            host_activation_handoff_permitted: false,
            runtime_boundary: "direct verified-reader static-control staging for the bounded layer-0/position-1 P7 diagnostic; no Engine, causal loop, HCLI, generation, or TPS claim",
        };
        Ok((source, ffn_norm, [hc_fn, hc_base, hc_scale]))
    }

    fn stage_verified_full(
        reader: &DeepSeekV4FullStreamReader,
        name: &str,
    ) -> ProbeResult<DeepSeekV4StagedTensor> {
        let metadata = reader.tensor_metadata(name)?;
        let bytes = usize::try_from(metadata.bytes)
            .map_err(|_| format!("{name} bytes exceed host usize"))?;
        let payload = reader.read_verified_full(name, bytes)?;
        if payload.len() != bytes {
            return Err(format!("{name} verified reader returned an unexpected length").into());
        }
        Ok(DeepSeekV4StagedTensor {
            name: metadata.name.clone(),
            dtype: metadata.dtype.clone(),
            shape: metadata.shape.clone(),
            source_shard: metadata.source_shard.clone(),
            range: 0..metadata.bytes,
            bytes: payload,
        })
    }

    fn source_binding(staged: &DeepSeekV4StagedTensor) -> DeepSeekV4P7SourceTensorBinding {
        DeepSeekV4P7SourceTensorBinding {
            name: staged.name.clone(),
            dtype: staged.dtype.clone(),
            shape: staged.shape.clone(),
            bytes: staged.bytes.len(),
            sha256: sha256(&staged.bytes),
        }
    }

    fn required_hot_cache_bytes(
        reader: &DeepSeekV4FullStreamReader,
        route_ids: &[u32],
    ) -> ProbeResult<u64> {
        route_ids.iter().try_fold(0u64, |total, &expert| {
            let descriptor = resolve_expert_bundle(reader, ExpertBundleKey::new(0, expert as u16))?;
            total
                .checked_add(descriptor.payload_bytes)
                .ok_or_else(|| "P6 hot expert capacity overflow".into())
        })
    }

    fn read_gpu_bytes(buffer: &metal::Buffer, length: usize) -> ProbeResult<Vec<u8>> {
        if buffer.length() < length as u64 {
            return Err("Metal buffer is smaller than post-completion diagnostic readback".into());
        }
        Ok(unsafe { std::slice::from_raw_parts(buffer.contents() as *const u8, length).to_vec() })
    }

    fn read_gpu_u16(buffer: &metal::Buffer, count: usize) -> ProbeResult<Vec<u16>> {
        let raw = read_gpu_bytes(buffer, count * std::mem::size_of::<u16>())?;
        Ok(raw
            .chunks_exact(2)
            .map(|chunk| u16::from_le_bytes([chunk[0], chunk[1]]))
            .collect())
    }

    fn read_gpu_u32(buffer: &metal::Buffer, count: usize) -> ProbeResult<Vec<u32>> {
        let raw = read_gpu_bytes(buffer, count * std::mem::size_of::<u32>())?;
        Ok(raw
            .chunks_exact(4)
            .map(|chunk| u32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]))
            .collect())
    }

    fn read_gpu_f32(buffer: &metal::Buffer, count: usize) -> ProbeResult<Vec<f32>> {
        let raw = read_gpu_bytes(buffer, count * std::mem::size_of::<f32>())?;
        let values = raw
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

    fn diagnostic_u32(expected: &[u32], observed: &[u32]) -> Value {
        json!({
            "expected_sha256": sha256_u32(expected),
            "observed_sha256": sha256_u32(observed),
            "bit_exact": expected == observed,
            "expected": expected,
            "observed": observed,
        })
    }

    fn diagnostic_u16(expected: &[u16], observed: &[u16]) -> Value {
        json!({
            "expected_sha256": sha256_u16(expected),
            "observed_sha256": sha256_u16(observed),
            "bit_exact": expected == observed,
            "element_count": observed.len(),
        })
    }

    fn source_combine_order_u32(order: &[Layer0MoeCombineOrder]) -> ProbeResult<Vec<(u32, u32)>> {
        order
            .iter()
            .map(|entry| {
                Ok((
                    u32::try_from(entry.source_top_slot)
                        .map_err(|_| std::io::Error::other("source MoE top slot exceeds u32"))?,
                    u32::try_from(entry.expert_id)
                        .map_err(|_| std::io::Error::other("source MoE expert ID exceeds u32"))?,
                ))
            })
            .collect()
    }

    fn diagnostic_f32(expected: &[f32], observed: &[f32]) -> Value {
        let (mismatch_count, max_abs, max_rel) = f32_distance(expected, observed);
        json!({
            "expected_sha256": sha256_f32(expected),
            "observed_sha256": sha256_f32(observed),
            "bit_exact": expected.len() == observed.len() && expected == observed,
            "element_count": observed.len(),
            "bit_mismatch_count": mismatch_count,
            "max_abs": max_abs,
            "max_relative": max_rel,
        })
    }

    fn observed_bf16(observed: &[u16]) -> Value {
        json!({
            "observed_sha256": sha256_u16(observed),
            "element_count": observed.len(),
        })
    }

    fn p7_mhc_control_v21_bounds() -> Bounds {
        // Local mHC controls include one 16K-wide reduction.  These are the
        // same op-local V2.1 envelope used for the corresponding source
        // mHC authority rung: they gate relative L2, cosine, meaningful
        // scale, and exact ranked decisions while preserving the FP64
        // reference as the authority for both F32 candidates.
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
        // This scores declared BF16 store boundaries against independently
        // accumulated FP64 values. Exact storage equality is reported
        // separately and is never silently replaced by this continuous gate.
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

    fn f32_distance(expected: &[f32], observed: &[f32]) -> (usize, f64, f64) {
        let mut mismatches = expected.len().abs_diff(observed.len());
        let mut max_abs = 0.0f64;
        let mut max_rel = 0.0f64;
        for (&left, &right) in expected.iter().zip(observed) {
            if left.to_bits() != right.to_bits() {
                mismatches += 1;
            }
            let abs = (f64::from(left) - f64::from(right)).abs();
            max_abs = max_abs.max(abs);
            max_rel = max_rel.max(abs / f64::from(left).abs().max(1.0e-30));
        }
        (mismatches, max_abs, max_rel)
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

    fn sha256(bytes: &[u8]) -> String {
        format!("{:x}", Sha256::digest(bytes))
    }

    fn sha256_join(parts: &[&str]) -> String {
        let mut hash = Sha256::new();
        for part in parts {
            hash.update(part.as_bytes());
            hash.update([0]);
        }
        format!("{:x}", hash.finalize())
    }

    fn sha256_u16(values: &[u16]) -> String {
        let mut hash = Sha256::new();
        for value in values {
            hash.update(value.to_le_bytes());
        }
        format!("{:x}", hash.finalize())
    }

    fn sha256_u32(values: &[u32]) -> String {
        let mut hash = Sha256::new();
        for value in values {
            hash.update(value.to_le_bytes());
        }
        format!("{:x}", hash.finalize())
    }

    fn sha256_f32(values: &[f32]) -> String {
        let mut hash = Sha256::new();
        for value in values {
            hash.update(value.to_bits().to_le_bytes());
        }
        format!("{:x}", hash.finalize())
    }
}

#[cfg(target_os = "macos")]
fn main() -> Result<(), Box<dyn std::error::Error>> {
    macos::run()
}

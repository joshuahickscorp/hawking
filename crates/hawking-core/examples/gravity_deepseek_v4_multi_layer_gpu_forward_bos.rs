//! Real multi-layer Metal forward over contiguous BOS layers of DeepSeek-V4-Flash.
//!
//! Default layers: 0..=42 (full base body). Layer 0 uses the P3A/P4A
//! embed→attention path; layers 1..N use the parameterized BOS window-KV
//! attention executor. Every layer runs P7 mHC-FFN + P6 MoE.
//!
//! Layer schedule honesty:
//! - layers 0,1: ratio-0 (full growing-KV BOS specialization, valid_kv=1)
//! - even layers ≥2: ratio-4 with indexer; at BOS/pos0 compressed topk is empty
//!   (`end_pos // 4 == 0`), so window-only sparse is the exact source path
//! - odd layers ≥3: ratio-128; same BOS empty-compressed specialization
//! - layers 0..2: hash tid2eid MoE (pre-resident experts)
//! - layers ≥3: learned-bias two-phase MoE (route on device → load six experts
//!   → expert body; host reads selected IDs only for residency)
//!
//! Usage:
//!   cargo run -p hawking-core --example gravity_deepseek_v4_multi_layer_gpu_forward_bos -- \
//!     --artifact <full-43-layer-stream.gravity> \
//!     --out <receipt.json> \
//!     [--max-layer 42]

#[cfg(not(target_os = "macos"))]
fn main() -> Result<(), Box<dyn std::error::Error>> {
    Err(std::io::Error::other(
        "gravity_deepseek_v4_multi_layer_gpu_forward_bos requires macOS Metal",
    )
    .into())
}

#[cfg(target_os = "macos")]
mod macos {
    use hawking_core::gravity_deepseek_v4::DeepSeekV4FullStreamReader;
    use hawking_core::gravity_deepseek_v4_attention_device::{
        DeepSeekV4Ratio0AttentionDeviceExecutor, DSV4F_RATIO0_GROWING_KV_SPARSE_ATTENTION_KERNEL,
    };
    use hawking_core::gravity_deepseek_v4_bos_layer_attention_device::{
        expected_bos_compress_ratio, DeepSeekV4BosLayerAttentionDeviceExecutor,
        DeepSeekV4BosLayerChildDeviceInput, DSV4F_BOS_LAYER_ATTENTION_DISPATCHES,
    };
    use hawking_core::gravity_deepseek_v4_execution_context::{
        DeepSeekV4ExecutionContext, DeepSeekV4ExecutionContextConfig, DeepSeekV4SelectedRouteSet,
    };
    use hawking_core::gravity_deepseek_v4_expert_cache::{
        resolve_expert_bundle, DeepSeekV4ExpertBundleCache, ExpertBundleKey,
    };
    use hawking_core::gravity_deepseek_v4_final_head::{
        host_greedy_lm_head, host_merge_final_head_from_hc_bf16, read_hc_bf16_from_buffer,
    };
    use hawking_core::gravity_deepseek_v4_layer0_prefix::PREFIX_TOKEN_ID;
    use hawking_core::gravity_deepseek_v4_layer_plan::DeepSeekV4LayerDeviceCatalog;
    use hawking_core::gravity_deepseek_v4_layer_scheduler::{
        DeepSeekV4LayerPreparationScheduler, DeepSeekV4LayerPreparationStage,
    };
    use hawking_core::gravity_deepseek_v4_layer_source_anchors::DeepSeekV4LayerGateMode;
    use hawking_core::gravity_deepseek_v4_p3a_stage_sink::DeepSeekV4P3aMetalStageSink;
    use hawking_core::gravity_deepseek_v4_p6_device::{
        DeepSeekV4Layer0P6MetalExecutor, DSV4F_P6_DEVICE_COMMAND_BUFFERS,
        DSV4F_P6_DEVICE_DISPATCHES, DSV4F_P6_LEARNED_DEVICE_COMMAND_BUFFERS,
        DSV4F_P6_LEARNED_DEVICE_DISPATCHES,
    };
    use hawking_core::gravity_deepseek_v4_p7_composition::{
        DeepSeekV4P7AttentionDeviceState, DeepSeekV4P7FfnSourceContract,
        DeepSeekV4P7SourceTensorBinding,
    };
    use hawking_core::gravity_deepseek_v4_p7_device::{
        DeepSeekV4P7BoundedDeviceExecutor, DeepSeekV4P7DeviceOutput,
        DSV4F_P7_OWNED_COMMAND_BUFFERS, DSV4F_P7_OWNED_DEVICE_DISPATCHES,
    };
    use hawking_core::gravity_deepseek_v4_runtime_spine::{
        DeepSeekV4ControlProjection, DeepSeekV4StagedTensor,
    };
    use hawking_core::metal::MetalContext;
    use serde_json::json;
    use sha2::{Digest, Sha256};
    use std::error::Error;
    use std::fs::{self, File};
    use std::io::Write;
    use std::path::PathBuf;
    use std::time::Instant;

    const RECEIPT_SCHEMA: &str = "hawking.gravity.deepseek_v4.multi_layer_gpu_forward_bos.v1";
    const PARITY: &str = "NUMERIC_PARITY_V2_1_ONLY";
    const L0_BOS_ROUTE_IDS: [u16; 6] = [254, 222, 245, 200, 53, 35];
    /// Default deepest layer is the last base layer (0..42).
    const DEFAULT_MAX_LAYER: usize = 42;

    type ProbeResult<T> = Result<T, Box<dyn Error>>;

    struct Args {
        artifact: PathBuf,
        out: PathBuf,
        max_layer: usize,
    }

    #[derive(Default)]
    struct StageAccounting {
        metal_dispatches: usize,
        command_buffers: usize,
        cpu_visible_waits: usize,
        stages: Vec<serde_json::Value>,
    }

    impl StageAccounting {
        fn record(
            &mut self,
            name: &str,
            dispatches: usize,
            command_buffers: usize,
            waits: usize,
            notes: serde_json::Value,
        ) {
            self.metal_dispatches += dispatches;
            self.command_buffers += command_buffers;
            self.cpu_visible_waits += waits;
            self.stages.push(json!({
                "stage": name,
                "metal_dispatches": dispatches,
                "command_buffers": command_buffers,
                "cpu_visible_waits": waits,
                "notes": notes,
            }));
        }
    }

    pub fn run() -> ProbeResult<()> {
        let args = parse_args()?;
        if args.max_layer >= 43 {
            return Err("--max-layer must be in 0..42".into());
        }
        let wall = Instant::now();

        let catalog = DeepSeekV4LayerDeviceCatalog::admit(&DeepSeekV4FullStreamReader::admit(
            &args.artifact,
        )?)?;

        // Admit every layer we intend to run under the BOS full-layer contract.
        let mut layers_run = Vec::new();
        let mut stop_reason: Option<String> = None;
        for layer in 0..=args.max_layer {
            match catalog.plan(layer)?.require_bos_full_layer_device() {
                Ok(()) => layers_run.push(layer),
                Err(err) => {
                    stop_reason = Some(format!("layer {layer}: {err}"));
                    break;
                }
            }
        }
        if layers_run.is_empty() {
            return Err(format!(
                "no layers admitted under BOS full-layer device contract: {}",
                stop_reason.unwrap_or_else(|| "unknown".into())
            )
            .into());
        }
        let deepest = *layers_run.last().unwrap();

        let mut context = DeepSeekV4ExecutionContext::open(
            &args.artifact,
            DeepSeekV4ExecutionContextConfig::default(),
        )?;
        let prepared = context.prepare_decode_input(PREFIX_TOKEN_ID as u32)?;
        if prepared.token_id != PREFIX_TOKEN_ID as u32 || prepared.position != 0 {
            return Err("BOS preparation did not bind token 0 / position 0".into());
        }
        let route_set = DeepSeekV4SelectedRouteSet::new(L0_BOS_ROUTE_IDS)?;

        // ---- Layer 0 attention (P3A/P4A, growing-KV sparse) ----
        let l0_attn_exec = DeepSeekV4Ratio0AttentionDeviceExecutor::prepare(&catalog, 0, 0)?;
        if l0_attn_exec.sparse_attention_kernel() != DSV4F_RATIO0_GROWING_KV_SPARSE_ATTENTION_KERNEL
        {
            return Err("layer-0 plan must use production growing-KV sparse kernel".into());
        }
        let mut attention_sink =
            DeepSeekV4P3aMetalStageSink::new_for_verified_p4a_continuation(&context, &prepared)?;
        let mut attention_scheduler =
            DeepSeekV4LayerPreparationScheduler::new(&context, 0, route_set)?;
        let expected_attention_stages = [
            DeepSeekV4LayerPreparationStage::MhcAttentionControl,
            DeepSeekV4LayerPreparationStage::AttentionControl(DeepSeekV4ControlProjection::WqA),
            DeepSeekV4LayerPreparationStage::AttentionControl(DeepSeekV4ControlProjection::WqB),
            DeepSeekV4LayerPreparationStage::AttentionControl(DeepSeekV4ControlProjection::Wkv),
            DeepSeekV4LayerPreparationStage::AttentionControl(DeepSeekV4ControlProjection::WoA),
            DeepSeekV4LayerPreparationStage::AttentionControl(DeepSeekV4ControlProjection::WoB),
        ];
        for expected in expected_attention_stages {
            let step = attention_scheduler
                .execute_next_with_sink(&mut context, &mut attention_sink)?
                .ok_or("L0 attention scheduler ended early")?;
            if step.stage != expected {
                return Err("L0 attention scheduler produced an unexpected stage".into());
            }
        }
        let p4a_report = attention_sink.finish_p4a_continuation()?;
        let l0_attn_dispatches = p4a_report
            .p3a_counters
            .actual_gpu_dispatches
            .saturating_add(p4a_report.p4a_counters.actual_gpu_dispatches);
        let l0_attn_cbs = p4a_report
            .p3a_counters
            .actual_command_buffers
            .saturating_add(p4a_report.p4a_counters.actual_command_buffers);
        let l0_attn_waits = p4a_report
            .p3a_counters
            .actual_cpu_visible_waits
            .saturating_add(p4a_report.p4a_counters.actual_cpu_visible_waits);

        let metal = attention_sink.metal_context();
        let mut accounting = StageAccounting::default();
        accounting.record(
            "l0_attention_p3a_p4a_growing_kv",
            l0_attn_dispatches,
            l0_attn_cbs,
            l0_attn_waits,
            json!({
                "layer": 0,
                "compression": "ratio_0",
                "sparse_kernel": l0_attn_exec.sparse_attention_kernel(),
                "valid_kv_count": l0_attn_exec.growing_kv.valid_kv_count,
            }),
        );

        // ---- Layer 0 P7 + P6 ----
        let reader = context.spine().reader();
        let child = run_p7_p6(
            metal,
            reader,
            0,
            PREFIX_TOKEN_ID as u32,
            Some(L0_BOS_ROUTE_IDS.map(u32::from)),
            DeepSeekV4P7AttentionDeviceState::position0(
                metal,
                attention_sink.p4a_attention_output_buffer()?,
                0,
                PREFIX_TOKEN_ID as u32,
            )?,
            &mut accounting,
        )?;

        // ---- Layers 1..=deepest via general BOS attention + P7/P6 ----
        let mut prev_child = child;
        let mut learned_layers: Vec<usize> = Vec::new();
        let mut hash_layers: Vec<usize> = vec![0];
        for layer in 1..=deepest {
            let plan = catalog.plan(layer)?;
            plan.require_bos_full_layer_device()?;
            let compress = expected_bos_compress_ratio(layer);
            let mut attn =
                DeepSeekV4BosLayerAttentionDeviceExecutor::prepare(metal, reader, layer)?;
            if attn.source_bindings().layer != layer
                || attn.source_bindings().compress_ratio != compress
            {
                return Err(format!(
                    "layer {layer} BOS attention bindings disagree with the schedule"
                )
                .into());
            }
            let input =
                DeepSeekV4BosLayerChildDeviceInput::from_p7_position0_child(metal, &prev_child)?;
            let attn_out = attn.execute(metal, input)?;
            attn_out.validate()?;
            accounting.record(
                &format!("l{layer}_attention_bos_window_kv"),
                attn_out.actual_gpu_dispatches,
                attn_out.actual_command_buffers,
                attn_out.actual_cpu_visible_waits,
                json!({
                    "layer": layer,
                    "compression_ratio": compress,
                    "compression": plan.compression.as_str(),
                    "gate_mode": plan.gate_mode.as_str(),
                    "declared_dispatches": DSV4F_BOS_LAYER_ATTENTION_DISPATCHES,
                    "bos_window_only": true,
                    "compressed_topk_empty_at_bos": compress > 0,
                    "sparse_kernel": DSV4F_RATIO0_GROWING_KV_SPARSE_ATTENTION_KERNEL,
                    "valid_kv_count": 1,
                }),
            );

            let attention = attn_out.p7_attention_state(metal)?;
            match plan.gate_mode {
                DeepSeekV4LayerGateMode::HashTokenIdToExpertIds => hash_layers.push(layer),
                DeepSeekV4LayerGateMode::LearnedScoresWithSelectionBias => {
                    learned_layers.push(layer)
                }
            }
            prev_child = run_p7_p6(
                metal,
                reader,
                layer,
                PREFIX_TOKEN_ID as u32,
                if layer == 0 {
                    Some(L0_BOS_ROUTE_IDS.map(u32::from))
                } else {
                    None
                },
                attention,
                &mut accounting,
            )?;
        }

        if accounting.metal_dispatches == 0 {
            return Err("multi-layer GPU forward recorded zero Metal dispatches".into());
        }

        let full_body = deepest == 42 && layers_run.len() == 43;
        // Final head + greedy only after a complete 43-layer BOS body.
        let mut greedy_token: Option<serde_json::Value> = None;
        let mut greedy_token_id: Option<u32> = None;
        if full_body {
            let hc_bits = read_hc_bf16_from_buffer(&prev_child.child_hc_state_bf16)?;
            let merge = host_merge_final_head_from_hc_bf16(reader, &hc_bits)?;
            // Host-streamed lm_head (honest bootstrap; ~1 GB streamed once).
            let greedy = host_greedy_lm_head(reader, &merge.merged_f32)?;
            greedy_token_id = Some(greedy.token_id);
            greedy_token = Some(json!({
                "token_id": greedy.token_id,
                "logit": greedy.logit,
                "vocab_size": greedy.vocab_size,
                "lm_head_on_device": greedy.lm_head_on_device,
                "argmax_on_device": greedy.argmax_on_device,
                "metal_dispatches": greedy.metal_dispatches,
                "final_head_path": "host_f64_mhc_merge_rmsnorm_then_host_streamed_lm_head_greedy",
                "hc_mix_weights": merge.mix_weights_f32,
                "flat_rsqrt": merge.flat_rsqrt,
            }));
            accounting.record(
                "final_mhc_head_norm_lm_head_greedy",
                greedy.metal_dispatches,
                greedy.command_buffers,
                0,
                greedy_token.clone().unwrap(),
            );
        }

        let wall_ms = wall.elapsed().as_secs_f64() * 1e3;
        let status = if full_body && greedy_token_id.is_some() {
            "PASS_MULTI_LAYER_GPU_FORWARD_BOS_L0_L42_GREEDY_TOKEN".to_string()
        } else if full_body {
            "PASS_MULTI_LAYER_GPU_FORWARD_BOS_L0_L42".to_string()
        } else {
            format!("PASS_MULTI_LAYER_GPU_FORWARD_BOS_L0_L{deepest}")
        };
        let has_ratio4 = layers_run
            .iter()
            .any(|&l| expected_bos_compress_ratio(l) == 4);
        let has_ratio128 = layers_run
            .iter()
            .any(|&l| expected_bos_compress_ratio(l) == 128);
        let receipt = json!({
            "schema": RECEIPT_SCHEMA,
            "status": status,
            "artifact": {
                "path": args.artifact.display().to_string(),
                "manifest_seal_sha256": catalog.identity().manifest_seal_sha256,
                "repository": catalog.identity().repository,
                "revision": catalog.identity().revision,
            },
            "scope": {
                "layers": layers_run,
                "deepest_layer": deepest,
                "token_id": PREFIX_TOKEN_ID,
                "token_position": 0,
                "gate_mode_span": if learned_layers.is_empty() {
                    "hash_tid2eid_only".to_string()
                } else {
                    format!(
                        "hash_layers_{:?}_learned_layers_{:?}_span",
                        hash_layers, learned_layers.first().zip(learned_layers.last())
                    )
                },
                "sparse_attention_kernel": DSV4F_RATIO0_GROWING_KV_SPARSE_ATTENTION_KERNEL,
                "mhc_control_exp": "darwin_double_double_control_domain_general",
                "requested_max_layer": args.max_layer,
                "stop_reason": stop_reason,
            },
            "metal": {
                "metal_dispatches": accounting.metal_dispatches,
                "command_buffers": accounting.command_buffers,
                "cpu_visible_waits": accounting.cpu_visible_waits,
                "fallback": 0,
                "host_intermediate_handoff_between_stages": false,
                "host_route_id_readback_for_learned_residency": !learned_layers.is_empty(),
            },
            "stages": accounting.stages,
            "parity": {
                "classification": PARITY,
                "exact_storage": false,
                "reason": "composed multi-layer path inherits NumericParityV21Only until a sealed exact-storage e2e receipt is earned on the full residual chain",
            },
            "honesty": {
                "full_metal_multi_layer_forward": true,
                "full_43_layer_bos_body": full_body,
                "serve_endpoint_flipped": false,
                "greedy_token_produced": greedy_token_id.is_some(),
                "greedy_token": greedy_token,
                "ratio_0_layers": [0, 1],
                "ratio_4_bos_window_only": has_ratio4,
                "ratio_4_full_compressed_graph": false,
                "ratio_128_bos_window_only": has_ratio128,
                "ratio_128_full_compressed_graph": false,
                "ratio_128_status": if has_ratio128 {
                    "bos_window_attention_and_learned_moe_composed"
                } else {
                    "not_reached"
                },
                "learned_bias_gate_status": if learned_layers.is_empty() {
                    "not_required_for_selected_span".to_string()
                } else {
                    format!(
                        "two_phase_p6_composed_layers_{}_to_{}",
                        learned_layers.first().unwrap(),
                        learned_layers.last().unwrap()
                    )
                },
                "layer_schedule_note": "only base layers 0 and 1 are ratio-0; even layers >=2 are ratio-4; odd layers >=3 are ratio-128; BOS compressed topk empty",
                "deepest_full_layer": deepest,
                "hash_layers_run": hash_layers,
                "learned_layers_run": learned_layers,
                "final_head_note": if greedy_token_id.is_some() {
                    "host_f64_mhc_head_merge_rmsnorm_plus_host_streamed_lm_head; full device lm_head gemv optional follow-on"
                } else {
                    "not_run_requires_full_43_layer_body"
                },
            },
            "wall_time_ms": wall_ms,
            "final_child_layer": prev_child.layer,
            "final_child_retained": true,
            "greedy_token_id": greedy_token_id,
        });

        if let Some(parent) = args.out.parent() {
            fs::create_dir_all(parent)?;
        }
        let pretty = serde_json::to_string_pretty(&receipt)?;
        let mut file = File::create(&args.out)?;
        file.write_all(pretty.as_bytes())?;
        file.write_all(b"\n")?;
        let seal = format!("{:x}", Sha256::digest(pretty.as_bytes()));
        println!("{pretty}");
        println!("receipt_path: {}", args.out.display());
        println!("receipt_sha256: {seal}");
        println!(
            "metal_dispatches: {} command_buffers: {} cpu_visible_waits: {}",
            accounting.metal_dispatches, accounting.command_buffers, accounting.cpu_visible_waits
        );
        println!("deepest_layer: {deepest}");
        if let Some(tid) = greedy_token_id {
            println!("greedy_token_id: {tid}");
        }
        println!("status: {status}");
        println!("parity: {PARITY}");
        println!("wall_time_ms: {wall_ms:.1}");
        Ok(())
    }

    fn run_p7_p6(
        metal: &MetalContext,
        reader: &DeepSeekV4FullStreamReader,
        layer: usize,
        token_id: u32,
        pinned_routes: Option<[u32; 6]>,
        attention: DeepSeekV4P7AttentionDeviceState<'_>,
        accounting: &mut StageAccounting,
    ) -> ProbeResult<DeepSeekV4P7DeviceOutput> {
        let (source, ffn_norm, mhc_ffn) = stage_p7_controls(reader, layer, token_id, 0)?;
        let catalog = DeepSeekV4LayerDeviceCatalog::admit(reader)?;
        let plan = catalog.plan(layer)?;
        let learned = plan.gate_mode == DeepSeekV4LayerGateMode::LearnedScoresWithSelectionBias;

        // Hash: pre-size cache from known routes. Learned: minimal placeholder
        // cache (experts load mid-execute after on-device route).
        let mut cache = if learned {
            // Non-zero capacity required by cache ctor; unused at prepare.
            DeepSeekV4ExpertBundleCache::new(1, 0)?
        } else {
            let route_ids_u32 = if let Some(routes) = pinned_routes {
                routes
            } else {
                let tid2eid_name = format!("layers.{layer}.ffn.gate.tid2eid");
                let meta = reader.tensor_metadata(&tid2eid_name)?;
                let bytes = reader.read_verified_full(&tid2eid_name, meta.bytes as usize)?;
                read_tid2eid_row_u16(&bytes, token_id as usize)?.map(u32::from)
            };
            DeepSeekV4ExpertBundleCache::new(
                required_hot_cache_bytes(reader, layer as u16, &route_ids_u32)?,
                0,
            )?
        };

        let p6 =
            DeepSeekV4Layer0P6MetalExecutor::prepare_for_p7(metal, reader, &mut cache, &source)?;
        if !learned {
            let route_ids_u32 = if let Some(routes) = pinned_routes {
                routes
            } else {
                let tid2eid_name = format!("layers.{layer}.ffn.gate.tid2eid");
                let meta = reader.tensor_metadata(&tid2eid_name)?;
                let bytes = reader.read_verified_full(&tid2eid_name, meta.bytes as usize)?;
                read_tid2eid_row_u16(&bytes, token_id as usize)?.map(u32::from)
            };
            if p6.source_bindings().selected_expert_ids_top_slot_order != route_ids_u32 {
                return Err(format!("L{layer} P6 tid2eid plan differs from source BOS row").into());
            }
        } else if !p6.source_bindings().host_route_id_readback_for_residency {
            return Err(format!(
                "L{layer} learned P6 must declare host route-id readback for residency"
            )
            .into());
        }

        let mut p7 = DeepSeekV4P7BoundedDeviceExecutor::prepare(
            metal,
            source.clone(),
            &ffn_norm,
            &mhc_ffn,
            Box::new(p6),
        )?;
        let _ = metal.drain_trace();
        let _ = metal.drain_stats();
        let output = p7.execute_position0(attention)?;
        output.validate()?;

        // Read device route IDs for the receipt (diagnostic; not an activation handoff).
        let route_ids_u32 = read_route_ids_from_output(&output)?;

        let (p6_dispatches, p6_cbs) = if learned {
            (
                DSV4F_P6_LEARNED_DEVICE_DISPATCHES,
                DSV4F_P6_LEARNED_DEVICE_COMMAND_BUFFERS,
            )
        } else {
            (DSV4F_P6_DEVICE_DISPATCHES, DSV4F_P6_DEVICE_COMMAND_BUFFERS)
        };
        let moe_dispatches = DSV4F_P7_OWNED_DEVICE_DISPATCHES + p6_dispatches;
        let moe_cbs = DSV4F_P7_OWNED_COMMAND_BUFFERS + p6_cbs;
        accounting.record(
            &format!("l{layer}_p7_mhc_ffn_p6_moe_mhc_post"),
            moe_dispatches,
            moe_cbs,
            moe_cbs,
            json!({
                "layer": layer,
                "token_id": token_id,
                "token_position": 0,
                "gate_mode": plan.gate_mode.as_str(),
                "route_ids_top_slot": route_ids_u32,
                "p6_dispatches": p6_dispatches,
                "p7_owned_dispatches": DSV4F_P7_OWNED_DEVICE_DISPATCHES,
                "host_activation_handoff": false,
                "host_route_id_readback_for_residency": learned,
            }),
        );
        Ok(output)
    }

    fn read_route_ids_from_output(output: &DeepSeekV4P7DeviceOutput) -> ProbeResult<[u32; 6]> {
        let buf = &output.p6.route_ids_u32;
        let ptr = buf.contents() as *const u8;
        if ptr.is_null() {
            return Err("route_ids buffer contents null".into());
        }
        let bytes = unsafe { std::slice::from_raw_parts(ptr, 6 * 4) };
        let mut out = [0u32; 6];
        for (i, slot) in out.iter_mut().enumerate() {
            *slot = u32::from_le_bytes(bytes[i * 4..i * 4 + 4].try_into()?);
        }
        Ok(out)
    }

    fn stage_p7_controls(
        reader: &DeepSeekV4FullStreamReader,
        layer: usize,
        token_id: u32,
        token_position: usize,
    ) -> ProbeResult<(
        DeepSeekV4P7FfnSourceContract,
        DeepSeekV4StagedTensor,
        [DeepSeekV4StagedTensor; 3],
    )> {
        let ffn_norm = stage_full(reader, &format!("layers.{layer}.ffn_norm.weight"))?;
        let hc_fn = stage_full(reader, &format!("layers.{layer}.hc_ffn_fn"))?;
        let hc_base = stage_full(reader, &format!("layers.{layer}.hc_ffn_base"))?;
        let hc_scale = stage_full(reader, &format!("layers.{layer}.hc_ffn_scale"))?;
        let source = DeepSeekV4P7FfnSourceContract {
            layer,
            token_id,
            token_position,
            ffn_norm: binding(&ffn_norm),
            hc_ffn_fn: binding(&hc_fn),
            hc_ffn_base: binding(&hc_base),
            hc_ffn_scale: binding(&hc_scale),
            source_parent_retained: false,
            source_upload_required_before_execution: true,
            host_activation_handoff_permitted: false,
            runtime_boundary:
                "multi-layer BOS GPU forward static P7 controls; no Engine/HCLI/serve/TPS claim",
        };
        Ok((source, ffn_norm, [hc_fn, hc_base, hc_scale]))
    }

    fn stage_full(
        reader: &DeepSeekV4FullStreamReader,
        name: &str,
    ) -> ProbeResult<DeepSeekV4StagedTensor> {
        let metadata = reader.tensor_metadata(name)?;
        let bytes = usize::try_from(metadata.bytes)
            .map_err(|_| format!("{name} bytes exceed host usize"))?;
        let payload = reader.read_verified_full(name, bytes)?;
        if payload.len() != bytes {
            return Err(format!("{name} verified reader returned unexpected length").into());
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

    fn binding(staged: &DeepSeekV4StagedTensor) -> DeepSeekV4P7SourceTensorBinding {
        DeepSeekV4P7SourceTensorBinding {
            name: staged.name.clone(),
            dtype: staged.dtype.clone(),
            shape: staged.shape.clone(),
            bytes: staged.bytes.len(),
            sha256: format!("{:x}", Sha256::digest(&staged.bytes)),
        }
    }

    fn required_hot_cache_bytes(
        reader: &DeepSeekV4FullStreamReader,
        layer: u16,
        route_ids: &[u32],
    ) -> ProbeResult<u64> {
        route_ids.iter().try_fold(0u64, |total, &expert| {
            let descriptor =
                resolve_expert_bundle(reader, ExpertBundleKey::new(layer, expert as u16))?;
            total
                .checked_add(descriptor.payload_bytes)
                .ok_or_else(|| "expert hot capacity overflow".into())
        })
    }

    fn read_tid2eid_row_u16(table: &[u8], token_id: usize) -> ProbeResult<[u16; 6]> {
        let row_bytes = 6 * 8;
        let start = token_id
            .checked_mul(row_bytes)
            .ok_or("tid2eid row offset overflow")?;
        let end = start
            .checked_add(row_bytes)
            .ok_or("tid2eid row end overflow")?;
        if end > table.len() {
            return Err("tid2eid row exceeds table".into());
        }
        let mut out = [0u16; 6];
        for (i, slot) in out.iter_mut().enumerate() {
            let off = start + i * 8;
            let raw = i64::from_le_bytes(table[off..off + 8].try_into()?);
            if raw < 0 || raw > u16::MAX as i64 {
                return Err(format!("tid2eid expert id {raw} out of u16 range").into());
            }
            *slot = raw as u16;
        }
        Ok(out)
    }

    fn parse_args() -> ProbeResult<Args> {
        let mut artifact = None;
        let mut out = None;
        let mut max_layer = DEFAULT_MAX_LAYER;
        let mut args = std::env::args().skip(1);
        while let Some(arg) = args.next() {
            match arg.as_str() {
                "--artifact" => {
                    artifact = Some(PathBuf::from(args.next().ok_or("--artifact needs a path")?));
                }
                "--out" => {
                    out = Some(PathBuf::from(args.next().ok_or("--out needs a path")?));
                }
                "--max-layer" => {
                    max_layer = args
                        .next()
                        .ok_or("--max-layer needs a value")?
                        .parse()
                        .map_err(|_| "--max-layer must be an integer")?;
                }
                "--help" | "-h" => {
                    println!(
                        "gravity_deepseek_v4_multi_layer_gpu_forward_bos --artifact <path> --out <path> [--max-layer N]"
                    );
                    std::process::exit(0);
                }
                other => return Err(format!("unknown argument {other}").into()),
            }
        }
        Ok(Args {
            artifact: artifact.ok_or("--artifact is required")?,
            out: out.ok_or("--out is required")?,
            max_layer,
        })
    }
}

#[cfg(target_os = "macos")]
fn main() -> Result<(), Box<dyn std::error::Error>> {
    macos::run()
}

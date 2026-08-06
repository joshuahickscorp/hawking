//! Real multi-layer Metal forward: DeepSeek-V4-Flash layers 0 → 1 (BOS / pos0).
//!
//! Composes, on one caller-owned Metal context:
//!
//! ```text
//! L0: P3A pre → ratio-0 growing-KV attention → mHC-post
//!     → P7 mHC-ffn-pre → P6 hash-gate MoE → mHC-ffn-post → child
//! L1: child → mHC-attn-pre → ratio-0 growing-KV attention → mHC-post
//!     → P7 mHC-ffn-pre → P6 hash-gate MoE → mHC-ffn-post → child
//! ```
//!
//! Both layers are ratio-0 + hash-gate (the only full device-supported pair).
//! Seals a receipt with real `metal_dispatches > 0`. Parity is classified
//! honestly as NumericParityV21Only (not exact-storage). Not a serve token,
//! decoder runtime, HCLI endpoint, or TPS claim.
//!
//! Usage:
//!   cargo run -p hawking-core --example gravity_deepseek_v4_multi_layer_gpu_forward_l0_l1 -- \
//!     --artifact <full-43-layer-stream.gravity> \
//!     --out <receipt.json>

#[cfg(not(target_os = "macos"))]
fn main() -> Result<(), Box<dyn std::error::Error>> {
    Err(std::io::Error::other(
        "gravity_deepseek_v4_multi_layer_gpu_forward_l0_l1 requires macOS Metal",
    )
    .into())
}

#[cfg(target_os = "macos")]
mod macos {
    use hawking_core::gravity_deepseek_v4::DeepSeekV4FullStreamReader;
    use hawking_core::gravity_deepseek_v4_attention_device::{
        DeepSeekV4Ratio0AttentionDeviceExecutor, DSV4F_RATIO0_GROWING_KV_SPARSE_ATTENTION_KERNEL,
    };
    use hawking_core::gravity_deepseek_v4_execution_context::{
        DeepSeekV4ExecutionContext, DeepSeekV4ExecutionContextConfig, DeepSeekV4SelectedRouteSet,
    };
    use hawking_core::gravity_deepseek_v4_expert_cache::{
        resolve_expert_bundle, DeepSeekV4ExpertBundleCache, ExpertBundleKey,
    };
    use hawking_core::gravity_deepseek_v4_layer0_prefix::PREFIX_TOKEN_ID;
    use hawking_core::gravity_deepseek_v4_layer1_attention_device::{
        DeepSeekV4L1BosChildDeviceInput, DeepSeekV4Layer1BosAttentionDeviceExecutor,
        DSV4F_L1_BOS_ATTENTION_DISPATCHES,
    };
    use hawking_core::gravity_deepseek_v4_layer_plan::DeepSeekV4LayerDeviceCatalog;
    use hawking_core::gravity_deepseek_v4_layer_scheduler::{
        DeepSeekV4LayerPreparationScheduler, DeepSeekV4LayerPreparationStage,
    };
    use hawking_core::gravity_deepseek_v4_p3a_stage_sink::DeepSeekV4P3aMetalStageSink;
    use hawking_core::gravity_deepseek_v4_p6_device::{
        DeepSeekV4Layer0P6MetalExecutor, DSV4F_P6_DEVICE_COMMAND_BUFFERS,
        DSV4F_P6_DEVICE_DISPATCHES,
    };
    use hawking_core::gravity_deepseek_v4_p7_composition::{
        DeepSeekV4P7AttentionDeviceState, DeepSeekV4P7FfnSourceContract,
        DeepSeekV4P7SourceTensorBinding,
    };
    use hawking_core::gravity_deepseek_v4_p7_device::{
        DeepSeekV4P7BoundedDeviceExecutor, DSV4F_P7_OWNED_COMMAND_BUFFERS,
        DSV4F_P7_OWNED_DEVICE_DISPATCHES,
    };
    use hawking_core::gravity_deepseek_v4_runtime_spine::{
        DeepSeekV4ControlProjection, DeepSeekV4StagedTensor,
    };
    use serde_json::json;
    use sha2::{Digest, Sha256};
    use std::error::Error;
    use std::fs::{self, File};
    use std::io::Write;
    use std::path::PathBuf;
    use std::time::Instant;

    const RECEIPT_SCHEMA: &str =
        "hawking.gravity.deepseek_v4.multi_layer_gpu_forward_l0_l1.v1";
    const RECEIPT_STATUS: &str = "PASS_MULTI_LAYER_GPU_FORWARD_L0_L1";
    const PARITY: &str = "NUMERIC_PARITY_V2_1_ONLY";
    /// Pinned BOS tid2eid row for layer 0 (source hash table).
    const L0_BOS_ROUTE_IDS: [u16; 6] = [254, 222, 245, 200, 53, 35];

    type ProbeResult<T> = Result<T, Box<dyn Error>>;

    struct Args {
        artifact: PathBuf,
        out: PathBuf,
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
        let wall = Instant::now();

        let catalog = DeepSeekV4LayerDeviceCatalog::admit(
            &DeepSeekV4FullStreamReader::admit(&args.artifact)?,
        )?;
        catalog.plan(0)?.require_full_layer_device()?;
        catalog.plan(1)?.require_full_layer_device()?;
        let l0_attn_exec = DeepSeekV4Ratio0AttentionDeviceExecutor::prepare(&catalog, 0, 0)?;
        let l1_attn_exec = DeepSeekV4Ratio0AttentionDeviceExecutor::prepare(&catalog, 1, 0)?;
        if l0_attn_exec.sparse_attention_kernel() != DSV4F_RATIO0_GROWING_KV_SPARSE_ATTENTION_KERNEL
            || l1_attn_exec.sparse_attention_kernel()
                != DSV4F_RATIO0_GROWING_KV_SPARSE_ATTENTION_KERNEL
        {
            return Err("ratio-0 plans must use production growing-KV sparse kernel".into());
        }

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
        let l0_attn_host_handoff = p4a_report
            .p3a_counters
            .host_intermediate_handoff_bytes
            .saturating_add(p4a_report.p4a_counters.host_intermediate_handoff_bytes);

        let metal = attention_sink.metal_context();
        let mut accounting = StageAccounting::default();
        accounting.record(
            "l0_attention_p3a_p4a_growing_kv",
            l0_attn_dispatches,
            l0_attn_cbs,
            l0_attn_waits,
            json!({
                "sparse_kernel": l0_attn_exec.sparse_attention_kernel(),
                "valid_kv_count": l0_attn_exec.growing_kv.valid_kv_count,
                "host_intermediate_handoff_bytes": l0_attn_host_handoff,
                "p3a_gpu_dispatches": p4a_report.p3a_counters.actual_gpu_dispatches,
                "p4a_gpu_dispatches": p4a_report.p4a_counters.actual_gpu_dispatches,
            }),
        );

        // ---- Layer 0 P7 + P6 MoE ----
        let reader = context.spine().reader();
        let (l0_source, l0_ffn_norm, l0_mhc_ffn) =
            stage_p7_controls(reader, 0, PREFIX_TOKEN_ID as u32, 0)?;
        let l0_route_ids_u32 = L0_BOS_ROUTE_IDS.map(u32::from);
        let mut l0_cache = DeepSeekV4ExpertBundleCache::new(
            required_hot_cache_bytes(reader, 0, &l0_route_ids_u32)?,
            0,
        )?;
        let l0_p6 = DeepSeekV4Layer0P6MetalExecutor::prepare_for_p7(
            metal,
            reader,
            &mut l0_cache,
            &l0_source,
        )?;
        if l0_p6.source_bindings().selected_expert_ids_top_slot_order != l0_route_ids_u32 {
            return Err("L0 P6 tid2eid plan differs from pinned BOS row".into());
        }
        let mut l0_p7 = DeepSeekV4P7BoundedDeviceExecutor::prepare(
            metal,
            l0_source.clone(),
            &l0_ffn_norm,
            &l0_mhc_ffn,
            Box::new(l0_p6),
        )?;
        let _ = metal.drain_trace();
        let _ = metal.drain_stats();
        let l0_attention = DeepSeekV4P7AttentionDeviceState::position0(
            metal,
            attention_sink.p4a_attention_output_buffer()?,
            0,
            PREFIX_TOKEN_ID as u32,
        )?;
        let l0_output = l0_p7.execute_position0(l0_attention)?;
        l0_output.validate()?;
        let l0_moe_dispatches = DSV4F_P7_OWNED_DEVICE_DISPATCHES + DSV4F_P6_DEVICE_DISPATCHES;
        let l0_moe_cbs = DSV4F_P7_OWNED_COMMAND_BUFFERS + DSV4F_P6_DEVICE_COMMAND_BUFFERS;
        accounting.record(
            "l0_p7_mhc_ffn_p6_moe_mhc_post",
            l0_moe_dispatches,
            l0_moe_cbs,
            l0_moe_cbs, // one wait per owned command buffer
            json!({
                "layer": 0,
                "token_id": PREFIX_TOKEN_ID,
                "token_position": 0,
                "p6_dispatches": DSV4F_P6_DEVICE_DISPATCHES,
                "p7_owned_dispatches": DSV4F_P7_OWNED_DEVICE_DISPATCHES,
                "host_activation_handoff": false,
            }),
        );

        // ---- Layer 1 attention (growing-KV) ----
        let l1_child = DeepSeekV4L1BosChildDeviceInput::from_p7_position0_child(metal, &l0_output)?;
        let mut l1_attn = DeepSeekV4Layer1BosAttentionDeviceExecutor::prepare(metal, reader)?;
        if l1_attn
            .source_bindings()
            .controls
            .iter()
            .any(|c| !c.name.starts_with("layers.1."))
        {
            return Err("L1 attention staged a non-layer-1 control".into());
        }
        let l1_attn_out = l1_attn.execute(metal, l1_child)?;
        l1_attn_out.validate()?;
        accounting.record(
            "l1_attention_growing_kv",
            l1_attn_out.actual_gpu_dispatches,
            l1_attn_out.actual_command_buffers,
            l1_attn_out.actual_cpu_visible_waits,
            json!({
                "sparse_kernel": l1_attn_exec.sparse_attention_kernel(),
                "valid_kv_count": l1_attn_exec.growing_kv.valid_kv_count,
                "declared_dispatches": DSV4F_L1_BOS_ATTENTION_DISPATCHES,
                "host_intermediate_handoff_bytes": l1_attn_out.host_intermediate_handoff_bytes,
            }),
        );

        // ---- Layer 1 P7 + P6 MoE ----
        let (l1_source, l1_ffn_norm, l1_mhc_ffn) =
            stage_p7_controls(reader, 1, PREFIX_TOKEN_ID as u32, 0)?;
        // Resolve L1 BOS routes from the live tid2eid row (hash gate).
        let l1_tid2eid_name = "layers.1.ffn.gate.tid2eid";
        let l1_tid_meta = reader.tensor_metadata(l1_tid2eid_name)?;
        let l1_tid_bytes =
            reader.read_verified_full(l1_tid2eid_name, l1_tid_meta.bytes as usize)?;
        let l1_route_ids = read_tid2eid_row_u16(&l1_tid_bytes, PREFIX_TOKEN_ID as usize)?;
        let l1_route_ids_u32 = l1_route_ids.map(u32::from);
        let mut l1_cache = DeepSeekV4ExpertBundleCache::new(
            required_hot_cache_bytes(reader, 1, &l1_route_ids_u32)?,
            0,
        )?;
        let l1_p6 = DeepSeekV4Layer0P6MetalExecutor::prepare_for_p7(
            metal,
            reader,
            &mut l1_cache,
            &l1_source,
        )?;
        if l1_p6.source_bindings().selected_expert_ids_top_slot_order != l1_route_ids_u32 {
            return Err("L1 P6 tid2eid plan differs from source BOS row".into());
        }
        let mut l1_p7 = DeepSeekV4P7BoundedDeviceExecutor::prepare(
            metal,
            l1_source.clone(),
            &l1_ffn_norm,
            &l1_mhc_ffn,
            Box::new(l1_p6),
        )?;
        let _ = metal.drain_trace();
        let _ = metal.drain_stats();
        let l1_attention = l1_attn_out.p7_attention_state(metal)?;
        let l1_output = l1_p7.execute_position0(l1_attention)?;
        l1_output.validate()?;
        let l1_moe_dispatches = DSV4F_P7_OWNED_DEVICE_DISPATCHES + DSV4F_P6_DEVICE_DISPATCHES;
        let l1_moe_cbs = DSV4F_P7_OWNED_COMMAND_BUFFERS + DSV4F_P6_DEVICE_COMMAND_BUFFERS;
        accounting.record(
            "l1_p7_mhc_ffn_p6_moe_mhc_post",
            l1_moe_dispatches,
            l1_moe_cbs,
            l1_moe_cbs,
            json!({
                "layer": 1,
                "token_id": PREFIX_TOKEN_ID,
                "token_position": 0,
                "route_ids_top_slot": l1_route_ids_u32,
                "p6_dispatches": DSV4F_P6_DEVICE_DISPATCHES,
                "p7_owned_dispatches": DSV4F_P7_OWNED_DEVICE_DISPATCHES,
                "host_activation_handoff": false,
            }),
        );

        if accounting.metal_dispatches == 0 {
            return Err("multi-layer GPU forward recorded zero Metal dispatches".into());
        }

        let wall_ms = wall.elapsed().as_secs_f64() * 1e3;
        let receipt = json!({
            "schema": RECEIPT_SCHEMA,
            "status": RECEIPT_STATUS,
            "artifact": {
                "path": args.artifact.display().to_string(),
                "manifest_seal_sha256": catalog.identity().manifest_seal_sha256,
                "repository": catalog.identity().repository,
                "revision": catalog.identity().revision,
            },
            "scope": {
                "layers": [0, 1],
                "token_id": PREFIX_TOKEN_ID,
                "token_position": 0,
                "compression": "ratio_0_only",
                "gate_mode": "hash_tid2eid",
                "sparse_attention_kernel": DSV4F_RATIO0_GROWING_KV_SPARSE_ATTENTION_KERNEL,
                "mhc_control_exp": "darwin_double_double_control_domain_general",
            },
            "metal": {
                "metal_dispatches": accounting.metal_dispatches,
                "command_buffers": accounting.command_buffers,
                "cpu_visible_waits": accounting.cpu_visible_waits,
                "fallback": 0,
                "host_intermediate_handoff_between_stages": false,
            },
            "stages": accounting.stages,
            "parity": {
                "classification": PARITY,
                "exact_storage": false,
                "reason": "composed multi-layer path inherits NumericParityV21Only until a sealed exact-storage e2e receipt is earned on the full residual chain",
            },
            "honesty": {
                "full_metal_multi_layer_forward": true,
                "serve_endpoint_flipped": false,
                "greedy_token_produced": false,
                "ratio_4_128_status": "not_in_scope_layers_0_1_are_ratio_0",
                "learned_bias_gate_status": "not_required_layers_0_1_are_hash_gate",
            },
            "wall_time_ms": wall_ms,
            "l0_child_retained": true,
            "l1_child_retained": true,
            "l0_p4b_predecessor_parity": l0_output.p4b_predecessor_parity.as_str(),
            "l1_p4b_predecessor_parity": l1_output.p4b_predecessor_parity.as_str(),
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
        println!("status: {RECEIPT_STATUS}");
        println!("parity: {PARITY}");
        Ok(())
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
            runtime_boundary: "multi-layer L0->L1 GPU forward static P7 controls; no Engine/HCLI/serve/TPS claim",
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
        // I64[*,6] table; each entry is little-endian i64 expert id.
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
        let mut args = std::env::args().skip(1);
        while let Some(arg) = args.next() {
            match arg.as_str() {
                "--artifact" => {
                    artifact = Some(PathBuf::from(args.next().ok_or("--artifact needs a path")?));
                }
                "--out" => {
                    out = Some(PathBuf::from(args.next().ok_or("--out needs a path")?));
                }
                "--help" | "-h" => {
                    println!(
                        "gravity_deepseek_v4_multi_layer_gpu_forward_l0_l1 --artifact <path> --out <path>"
                    );
                    std::process::exit(0);
                }
                other => return Err(format!("unknown argument {other}").into()),
            }
        }
        Ok(Args {
            artifact: artifact.ok_or("--artifact is required")?,
            out: out.ok_or("--out is required")?,
        })
    }
}

#[cfg(target_os = "macos")]
fn main() -> Result<(), Box<dyn std::error::Error>> {
    macos::run()
}

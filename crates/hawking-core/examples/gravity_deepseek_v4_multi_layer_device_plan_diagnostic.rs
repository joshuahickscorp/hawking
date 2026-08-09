//! Multi-layer DeepSeek-V4 device-plan + staging diagnostic.
//!
//! Walks layers 0, 1, and at least one later layer (default 2) through:
//! 1. verified layer-source anchors / device catalog;
//! 2. ratio-0 attention plan resolution (or honest refusal);
//! 3. MoE/gate plan resolution (or honest refusal);
//! 4. scheduler staging with the verified tensor cache so static controls are
//!    not re-streamed between layers.
//!
//! This is deliberately *not* a full multi-layer Metal forward receipt. It
//! records which layers the parameterized device surface admits and proves
//! the staging/cache path across >=3 layers. Metal dispatches remain zero
//! unless a future sibling diagnostic attaches a real device sink.
//!
//! Usage:
//!   cargo run -p hawking-core --example gravity_deepseek_v4_multi_layer_device_plan_diagnostic -- \
//!     --artifact <full-43-layer-stream.gravity> --out <receipt.json>

use std::error::Error;
use std::fs::{self, File};
use std::io::Write;
use std::path::PathBuf;

use hawking_core::gravity_deepseek_v4_attention_device::DeepSeekV4Ratio0AttentionDevicePlan;
use hawking_core::gravity_deepseek_v4_execution_context::{
    DeepSeekV4ControlPayload, DeepSeekV4ExecutionContext, DeepSeekV4ExecutionContextConfig,
    DeepSeekV4SelectedRouteSet,
};
use hawking_core::gravity_deepseek_v4_layer_plan::DeepSeekV4LayerDeviceCatalog;
use hawking_core::gravity_deepseek_v4_layer_scheduler::{
    DeepSeekV4LayerPreparationScheduler, DeepSeekV4NativeStage, DeepSeekV4NativeStageConsumption,
    DeepSeekV4NativeStageSink,
};
use hawking_core::Result as CoreResult;
use serde_json::json;
use sha2::{Digest, Sha256};

const RECEIPT_SCHEMA: &str = "hawking.gravity.deepseek_v4.multi_layer_device_plan_diagnostic.v1";
const RECEIPT_STATUS: &str = "PASS_MULTI_LAYER_DEVICE_PLAN_AND_STAGING_NOT_FULL_METAL_FORWARD";
const ONE_EXPERT_BUNDLE_BYTES: u64 = 13_369_344;

type ExampleResult<T> = Result<T, Box<dyn Error>>;

struct Args {
    artifact: PathBuf,
    out: PathBuf,
    layers: Vec<usize>,
}

#[derive(Default)]
struct CountingStagingSink {
    stages: usize,
    control_bytes: usize,
    metal_dispatches: usize,
}

impl DeepSeekV4NativeStageSink for CountingStagingSink {
    fn consume_native_stage(
        &mut self,
        stage: DeepSeekV4NativeStage<'_>,
    ) -> CoreResult<DeepSeekV4NativeStageConsumption> {
        self.stages += 1;
        match stage {
            DeepSeekV4NativeStage::Control { payload, .. } => {
                self.control_bytes += match payload {
                    DeepSeekV4ControlPayload::EmbeddingRow { bf16_bits, .. } => bf16_bits.len() * 2,
                    DeepSeekV4ControlPayload::Tensor(tensor) => tensor.bytes.len(),
                    DeepSeekV4ControlPayload::NativePair(pair) => {
                        pair.weight.bytes.len() + pair.scale.bytes.len()
                    }
                    DeepSeekV4ControlPayload::MhcControl { tensors, .. } => {
                        tensors.iter().map(|t| t.bytes.len()).sum()
                    }
                };
            }
            DeepSeekV4NativeStage::RoutedExpertWave { .. } => {}
        }
        // Explicit zero device work: this diagnostic is staging/plan only.
        self.metal_dispatches = 0;
        Ok(DeepSeekV4NativeStageConsumption::default())
    }
}

fn main() -> ExampleResult<()> {
    let args = parse_args()?;
    let catalog = DeepSeekV4LayerDeviceCatalog::admit(
        &hawking_core::gravity_deepseek_v4::DeepSeekV4FullStreamReader::admit(&args.artifact)?,
    )?;
    let full_supported = catalog.full_layer_device_supported();
    let attention_supported = catalog.attention_device_supported();

    let mut layer_reports = Vec::new();
    for &layer in &args.layers {
        let plan = catalog.plan(layer)?;
        let attention_plan = DeepSeekV4Ratio0AttentionDevicePlan::resolve(&catalog, layer, 0);
        layer_reports.push(json!({
            "layer": layer,
            "compression": plan.compression.as_str(),
            "compression_ratio": plan.compression.ratio(),
            "gate_mode": plan.gate_mode.as_str(),
            "attention_device_supported": plan.attention_device_supported,
            "moe_device_supported": plan.moe_device_supported,
            "attention_refusal": plan.attention_refusal,
            "moe_refusal": plan.moe_refusal,
            "mhc_control_exp": plan.mhc_control_exp.as_str(),
            "ratio0_attention_plan": match attention_plan {
                Ok(step) => json!({
                    "status": "resolved",
                    "token_position": step.token_position,
                    "valid_kv_count": step.valid_kv_count,
                    "sparse_attention_kernel": step.sparse_attention_kernel,
                    "hc_attn_fn": step.tensor_names.hc_attn_fn,
                    "wq_a_weight": step.tensor_names.wq_a_weight,
                }),
                Err(err) => json!({
                    "status": "refused",
                    "error": err.to_string(),
                }),
            },
        }));
    }

    let config = DeepSeekV4ExecutionContextConfig {
        routed_expert_hot_capacity_bytes: ONE_EXPERT_BUNDLE_BYTES * 6,
        routed_expert_cold_capacity_bytes: ONE_EXPERT_BUNDLE_BYTES,
        ..DeepSeekV4ExecutionContextConfig::default()
    };
    let mut context = DeepSeekV4ExecutionContext::open(&args.artifact, config)?;
    let anchors = context.layer_source_anchors().clone();
    let _prepared = context.prepare_decode_input(0)?;
    let route_set = DeepSeekV4SelectedRouteSet::new([0, 1, 2, 3, 4, 5])?;
    let mut staging_reports = Vec::new();
    let mut total_stages = 0usize;
    let mut total_control_bytes = 0usize;

    for &layer in &args.layers {
        // Only stage layers the scheduler can accept topologically. The
        // scheduler itself is layer-general; MoE/attention device support is
        // recorded separately above.
        let mut scheduler = DeepSeekV4LayerPreparationScheduler::new(&context, layer, route_set)?;
        let mut sink = CountingStagingSink::default();
        while scheduler
            .execute_next_with_sink(&mut context, &mut sink)?
            .is_some()
        {}
        if !scheduler.is_complete() {
            return Err(failure(format!(
                "layer {layer} scheduler did not complete its staging program"
            )));
        }
        total_stages += sink.stages;
        total_control_bytes += sink.control_bytes;
        let cache_counters = context.verified_tensor_cache_counters();
        staging_reports.push(json!({
            "layer": layer,
            "scheduler_stages": sink.stages,
            "control_bytes_consumed_by_sink": sink.control_bytes,
            "metal_dispatches": sink.metal_dispatches,
            "verified_tensor_cache": cache_counters.map(|c| json!({
                "requests": c.requests,
                "hits": c.hits,
                "misses": c.misses,
                "verified_source_reads": c.verified_source_reads,
                "evictions": c.evictions,
            })),
        }));
        // Reset mHC so the next layer can be scheduled independently for this
        // staging-only diagnostic (not a true residual chain).
        context.reset_decode_state();
        let _ = context.prepare_decode_input(0)?;
    }

    let cache_final = context.verified_tensor_cache_counters();
    let receipt = json!({
        "schema": RECEIPT_SCHEMA,
        "status": RECEIPT_STATUS,
        "artifact": {
            "path": args.artifact.display().to_string(),
            "manifest_seal_sha256": anchors.identity().manifest_seal_sha256,
            "repository": anchors.identity().repository,
            "revision": anchors.identity().revision,
        },
        "layers_requested": args.layers,
        "full_layer_device_supported": full_supported,
        "attention_device_supported": attention_supported,
        "mhc_control_exp_promoted": "darwin_double_double_control_domain_general",
        "layer_reports": layer_reports,
        "staging": {
            "layers": staging_reports,
            "total_scheduler_stages": total_stages,
            "total_control_bytes_consumed_by_sink": total_control_bytes,
            "metal_dispatches": 0,
            "verified_tensor_cache_final": cache_final.map(|c| json!({
                "requests": c.requests,
                "hits": c.hits,
                "misses": c.misses,
                "verified_source_reads": c.verified_source_reads,
                "evictions": c.evictions,
                "source_payload_bytes": c.source_payload_bytes,
            })),
        },
        "honesty": {
            "full_metal_multi_layer_forward": false,
            "serve_endpoint_flipped": false,
            "parity_classification": "staging_and_plan_only_no_numeric_parity_claim",
            "ratio_4_128_status": "full_growing_kv_refused_non_bos; bos_window_admitted",
            "learned_bias_gate_status": "two_phase_p6_admitted_for_bos_full_layer",
        },
    });

    if let Some(parent) = args.out.parent() {
        fs::create_dir_all(parent)?;
    }
    let mut file = File::create(&args.out)?;
    file.write_all(serde_json::to_string_pretty(&receipt)?.as_bytes())?;
    file.write_all(b"\n")?;
    println!("{}", serde_json::to_string_pretty(&receipt)?);
    let _ = sha256(serde_json::to_string(&receipt)?.as_bytes());
    Ok(())
}

fn parse_args() -> ExampleResult<Args> {
    let mut artifact = None;
    let mut out = None;
    let mut layers = vec![0usize, 1, 2];
    let mut args = std::env::args().skip(1);
    while let Some(arg) = args.next() {
        match arg.as_str() {
            "--artifact" => {
                artifact = Some(PathBuf::from(args.next().ok_or("--artifact needs a path")?));
            }
            "--out" => {
                out = Some(PathBuf::from(args.next().ok_or("--out needs a path")?));
            }
            "--layers" => {
                let raw = args.next().ok_or("--layers needs a comma-separated list")?;
                layers = raw
                    .split(',')
                    .map(|part| part.trim().parse::<usize>())
                    .collect::<Result<Vec<_>, _>>()?;
                if layers.len() < 3 {
                    return Err(failure("--layers must list at least three layers"));
                }
            }
            other => return Err(failure(format!("unknown argument {other}"))),
        }
    }
    Ok(Args {
        artifact: artifact.ok_or("--artifact is required")?,
        out: out.ok_or("--out is required")?,
        layers,
    })
}

fn failure(message: impl Into<String>) -> Box<dyn Error> {
    std::io::Error::new(std::io::ErrorKind::InvalidData, message.into()).into()
}

fn sha256(bytes: &[u8]) -> String {
    format!("{:x}", Sha256::digest(bytes))
}

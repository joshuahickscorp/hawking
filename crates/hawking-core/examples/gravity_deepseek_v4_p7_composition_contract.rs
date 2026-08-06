//! Inspect the non-executing P7 source/device composition contract.
//!
//! This example intentionally has no `--out` argument and writes no receipt.
//! It validates that the real bounded mHC-FFN source lease can be consumed
//! while live, then prints metadata only.  It does not create Metal buffers,
//! execute kernels, run P4B/P6, or make a runtime/parity/TPS claim.

use std::error::Error;
use std::path::PathBuf;

use hawking_core::gravity_deepseek_v4_execution_context::{
    DeepSeekV4ExecutionContext, DeepSeekV4ExecutionContextConfig, DeepSeekV4SelectedRouteSet,
};
use hawking_core::gravity_deepseek_v4_layer_scheduler::{
    DeepSeekV4LayerPreparationResult, DeepSeekV4LayerPreparationScheduler,
    DeepSeekV4LayerPreparationStage,
};
use hawking_core::gravity_deepseek_v4_p7_composition::DeepSeekV4P7SourceLeasePreparation;
use serde_json::json;

type ExampleResult<T> = Result<T, Box<dyn Error>>;

struct Args {
    artifact: PathBuf,
}

fn main() -> ExampleResult<()> {
    let args = parse_args()?;
    let mut context = DeepSeekV4ExecutionContext::open(
        &args.artifact,
        DeepSeekV4ExecutionContextConfig::default(),
    )?;
    let prepared = context.prepare_decode_input(0)?;
    let mut preparation = DeepSeekV4P7SourceLeasePreparation::new(&context, &prepared, 0)?;
    let routes = DeepSeekV4SelectedRouteSet::new([0, 1, 2, 3, 4, 5])?;
    let mut scheduler = DeepSeekV4LayerPreparationScheduler::new(&context, 0, routes)?;
    let step = loop {
        let step = scheduler
            .execute_next(&mut context)?
            .ok_or_else(|| failure("scheduler ended before MhcFfnControl"))?;
        if step.stage == DeepSeekV4LayerPreparationStage::MhcFfnControl {
            break step;
        }
    };
    let lease = match &step.result {
        DeepSeekV4LayerPreparationResult::ControlLease(lease) => *lease,
        DeepSeekV4LayerPreparationResult::RoutedExpertAccesses(_) => {
            return Err(failure("MhcFfnControl did not return a control lease"));
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
        return Err(failure(
            "P7 preparation unexpectedly reported execution or host activation handoff",
        ));
    }
    let contract = preparation.source_contract()?;
    let full_causal_execution_denied = context.require_full_causal_execution().is_err();
    if !full_causal_execution_denied {
        return Err(failure(
            "P7 preparation unexpectedly admitted full causal execution",
        ));
    }
    println!(
        "{}",
        serde_json::to_string_pretty(&json!({
            "state": "P7_SOURCE_LEASE_INSPECTED_NOT_EXECUTABLE",
            "source_contract": {
                "layer": contract.layer,
                "token_id": contract.token_id,
                "token_position": contract.token_position,
                "ffn_norm": binding_json(&contract.ffn_norm),
                "hc_ffn_fn": binding_json(&contract.hc_ffn_fn),
                "hc_ffn_base": binding_json(&contract.hc_ffn_base),
                "hc_ffn_scale": binding_json(&contract.hc_ffn_scale),
                "source_parent_retained": contract.source_parent_retained,
                "source_upload_required_before_execution": contract.source_upload_required_before_execution,
                "host_activation_handoff_permitted": contract.host_activation_handoff_permitted,
            },
            "scheduler": {
                "actual_mhc_ffn_sequence": step.sequence,
                "actual_mhc_ffn_graph_node_ordinal": step.logical_graph_node_ordinal,
                "scheduler_complete": scheduler.is_complete(),
                "next_stage": scheduler.next_stage().map(|stage| stage.as_str()),
            },
            "execution": {
                "metal_buffers_created": 0,
                "command_buffers": 0,
                "compute_encoders": 0,
                "gpu_dispatches": 0,
                "cpu_visible_waits": 0,
                "host_intermediate_handoff_bytes": 0,
                "p4b_integrated": false,
                "p6_integrated": false,
                "numeric_parity_v21": false,
                "engine": false,
                "hcli": false,
                "base_true_tps": false,
                "full_causal_execution_denied": full_causal_execution_denied,
                "boundary": contract.runtime_boundary,
            },
        }))?
    );
    Ok(())
}

fn binding_json(
    binding: &hawking_core::gravity_deepseek_v4_p7_composition::DeepSeekV4P7SourceTensorBinding,
) -> serde_json::Value {
    json!({
        "name": binding.name,
        "dtype": binding.dtype,
        "shape": binding.shape,
        "bytes": binding.bytes,
        "sha256": binding.sha256,
    })
}

fn parse_args() -> ExampleResult<Args> {
    let mut artifact = None;
    let mut args = std::env::args().skip(1);
    while let Some(argument) = args.next() {
        match argument.as_str() {
            "--artifact" => artifact = args.next().map(PathBuf::from),
            "--help" | "-h" => {
                println!(
                    "usage: gravity_deepseek_v4_p7_composition_contract --artifact <absolute full Gravity dir>"
                );
                std::process::exit(0);
            }
            other => return Err(failure(format!("unknown argument {other:?}"))),
        }
    }
    let artifact = artifact.ok_or_else(|| failure("--artifact is required"))?;
    if !artifact.is_absolute() {
        return Err(failure("--artifact must be an absolute path"));
    }
    Ok(Args { artifact })
}

fn failure(message: impl Into<String>) -> Box<dyn Error> {
    std::io::Error::new(std::io::ErrorKind::InvalidData, message.into()).into()
}

#![cfg(target_os = "macos")]
use hawking_core::cost_ledger::{self, TokenCostReport};
use hawking_core::gravity_glm::gpu::GravityGlmGpu;
use hawking_core::gravity_glm::{
    GlmTrace, GPU_COMPACT_ATTENTION_ICB_ENV, GPU_COMPACT_MLA_ENV, GPU_DEVICE_DSA_ENV,
    GPU_DEVICE_ROUTER_ENV, GPU_EXPERT_TABLE_HIT_ENV, GPU_EXPERT_TABLE_ICB_ENV, GPU_EXPERT_WAVE_ENV,
    GPU_LM_HEAD_ENV, GPU_LM_HEAD_FULL_LOGITS_ENV, GPU_LM_HEAD_ICB_ENV,
};
use hawking_core::metal::MetalContext;
use std::path::PathBuf;
#[derive(Clone, Copy)]
struct ReplayMode {
    name: &'static str,
    attention: bool,
    expert: bool,
    head: bool,
}
const REPLAY_MODES: [ReplayMode; 5] = [
    ReplayMode {
        name: "direct",
        attention: false,
        expert: false,
        head: false,
    },
    ReplayMode {
        name: "attention_only",
        attention: true,
        expert: false,
        head: false,
    },
    ReplayMode {
        name: "expert_only",
        attention: false,
        expert: true,
        head: false,
    },
    ReplayMode {
        name: "head_only",
        attention: false,
        expert: false,
        head: true,
    },
    ReplayMode {
        name: "all_replay",
        attention: true,
        expert: true,
        head: true,
    },
];
fn set_replay_mode(mode: ReplayMode) {
    for (flag, enabled) in [
        (GPU_COMPACT_ATTENTION_ICB_ENV, mode.attention),
        (GPU_EXPERT_TABLE_ICB_ENV, mode.expert),
        (GPU_LM_HEAD_ICB_ENV, mode.head),
    ] {
        if enabled {
            std::env::set_var(flag, "1");
        } else {
            std::env::remove_var(flag);
        }
    }
}
fn percentile(values: &[u64], percentile: usize) -> u64 {
    let mut sorted = values.to_vec();
    sorted.sort_unstable();
    let rank = percentile
        .saturating_mul(sorted.len().saturating_sub(1))
        .saturating_add(99)
        / 100;
    sorted[rank.min(sorted.len().saturating_sub(1))]
}
fn bucket(report: &TokenCostReport, name: &str) -> u64 {
    report
        .buckets_us
        .get(name)
        .unwrap_or_else(|| {
            panic!(
                "missing {name} bucket; available={:?}",
                report.buckets_us.keys().collect::<Vec<_>>()
            )
        })
        .as_u64()
        .unwrap_or_else(|| panic!("missing {name} bucket"))
}
fn profiled_forward(
    model: &GravityGlmGpu,
    tokens: &[u32],
    mode: ReplayMode,
) -> (Vec<f32>, GlmTrace, TokenCostReport, u64) {
    set_replay_mode(mode);
    let _ = cost_ledger::end_token();
    assert!(cost_ledger::begin_token());
    let (logits, trace) = model.forward(tokens).expect("bounded profiled forward");
    let report = cost_ledger::end_token().expect("bounded token cost report");
    let waits = model
        .last_resident_waits()
        .expect("bounded resident wait count");
    (logits, trace, report, waits)
}
#[test]
#[ignore = "explicit bounded complete-token Metal replay benchmark"]
fn compact_glm_all_icb_flags_complete_token_encode_benchmark() {
    const SAMPLES: usize = 101;
    let Some(dir) = std::env::var_os("HAWKING_GLM_COMPACT_FIXTURE_DIR").map(PathBuf::from) else {
        panic!("set HAWKING_GLM_COMPACT_FIXTURE_DIR to the bounded direct-u8 compact fixture");
    };
    let authority: serde_json::Value = serde_json::from_slice(
        &std::fs::read(dir.join("ref_logits_f64.json")).expect("bounded FP64 authorities"),
    )
    .expect("parse bounded FP64 authorities");
    let tokens: Vec<u32> = authority[1]["tokens"]
        .as_array()
        .expect("one-token authority")
        .iter()
        .map(|value| value.as_u64().expect("token id") as u32)
        .collect();
    assert_eq!(tokens.len(), 1);
    for flag in [
        GPU_COMPACT_MLA_ENV,
        GPU_DEVICE_DSA_ENV,
        GPU_DEVICE_ROUTER_ENV,
        GPU_EXPERT_WAVE_ENV,
        GPU_EXPERT_TABLE_HIT_ENV,
        GPU_LM_HEAD_ENV,
        GPU_LM_HEAD_FULL_LOGITS_ENV,
    ] {
        std::env::set_var(flag, "1");
    }
    set_replay_mode(REPLAY_MODES[0]);
    let ctx = MetalContext::new().expect("Metal context");
    let model =
        GravityGlmGpu::open_dir_with_budget_resident(ctx, &dir, true, 512 * 1024 * 1024, true)
            .expect("bounded compact resident model");
    for mode in REPLAY_MODES {
        for _ in 0..3 {
            set_replay_mode(mode);
            model.forward(&tokens).expect("warm replay mode");
        }
    }
    set_replay_mode(REPLAY_MODES[0]);
    let (reference_logits, reference_trace) = model.forward(&tokens).expect("direct reference");
    let reference_waits = model.last_resident_waits().expect("direct reference waits");
    cost_ledger::set_enabled(true);
    let mut reports: Vec<Vec<TokenCostReport>> = REPLAY_MODES
        .iter()
        .map(|_| Vec::with_capacity(SAMPLES))
        .collect();
    for iteration in 0..SAMPLES {
        for offset in 0..REPLAY_MODES.len() {
            let mode_index = if iteration % 2 == 0 {
                (iteration + offset) % REPLAY_MODES.len()
            } else {
                (iteration + REPLAY_MODES.len() - offset) % REPLAY_MODES.len()
            };
            let mode = REPLAY_MODES[mode_index];
            let (logits, trace, report, waits) = profiled_forward(&model, &tokens, mode);
            assert_eq!(
                logits
                    .iter()
                    .map(|value| value.to_bits())
                    .collect::<Vec<_>>(),
                reference_logits
                    .iter()
                    .map(|value| value.to_bits())
                    .collect::<Vec<_>>(),
                "mode={} changed complete-token logits",
                mode.name
            );
            assert_eq!(
                trace.final_topk, reference_trace.final_topk,
                "mode={} changed DSA rank",
                mode.name
            );
            assert_eq!(
                trace.expert_choices, reference_trace.expert_choices,
                "mode={} changed expert choices",
                mode.name
            );
            assert_eq!(
                trace.sample_token, reference_trace.sample_token,
                "mode={} changed sampled token",
                mode.name
            );
            assert_eq!(waits, reference_waits, "mode={} changed waits", mode.name);
            reports[mode_index].push(report);
        }
    }
    cost_ledger::set_enabled(false);
    set_replay_mode(REPLAY_MODES[0]);
    let summarize = |reports: &[TokenCostReport]| {
        let wall: Vec<u64> = reports.iter().map(|report| report.wall_us).collect();
        let encode: Vec<u64> = reports
            .iter()
            .map(|report| bucket(report, "metal_encode"))
            .collect();
        let submit: Vec<u64> = reports
            .iter()
            .map(|report| bucket(report, "metal_submit"))
            .collect();
        let synchronize: Vec<u64> = reports
            .iter()
            .map(|report| bucket(report, "metal_synchronize_cpu_wait"))
            .collect();
        let gpu: Vec<u64> = reports
            .iter()
            .map(|report| report.device.gpu_execution_us)
            .collect();
        serde_json::json!({
            "samples": reports.len(),
            "wall_us": {"p50": percentile(&wall, 50), "p95": percentile(&wall, 95)},
            "metal_encode_us": {"p50": percentile(&encode, 50), "p95": percentile(&encode, 95)},
            "metal_submit_us": {"p50": percentile(&submit, 50), "p95": percentile(&submit, 95)},
            "metal_synchronize_us": {
                "p50": percentile(&synchronize, 50),
                "p95": percentile(&synchronize, 95)
            },
            "gpu_execution_us": {"p50": percentile(&gpu, 50), "p95": percentile(&gpu, 95)},
            "dispatches": {
                "p50": percentile(
                    &reports
                        .iter()
                        .map(|report| report.counters.dispatches_encoded)
                        .collect::<Vec<_>>(),
                    50
                )
            },
            "command_buffers": {
                "p50": percentile(
                    &reports
                        .iter()
                        .map(|report| report.counters.command_buffers_submitted)
                        .collect::<Vec<_>>(),
                    50
                )
            },
            "synchronization_points": {
                "p50": percentile(
                    &reports
                        .iter()
                        .map(|report| report.counters.synchronization_points)
                        .collect::<Vec<_>>(),
                    50
                )
            },
            "allocations": {
                "p50": percentile(
                    &reports
                        .iter()
                        .map(|report| report.counters.allocations)
                        .collect::<Vec<_>>(),
                    50
                )
            }
        })
    };
    let mut modes = serde_json::Map::new();
    for (mode, reports) in REPLAY_MODES.iter().zip(&reports) {
        modes.insert(mode.name.to_string(), summarize(reports));
    }
}

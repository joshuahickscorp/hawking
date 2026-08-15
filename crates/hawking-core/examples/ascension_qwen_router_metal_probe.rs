//! Direct Metal router probe for the two Ascension Qwen manager candidates.
//!
//! This is intentionally a component probe: it validates one routing launch
//! against a CPU oracle for each exact expert/top-k shape.  It does not load
//! model weights, run a decoder, establish a `.gravity` artifact, or provide
//! any model TPS evidence.

#[cfg(not(target_os = "macos"))]
fn main() -> Result<(), Box<dyn std::error::Error>> {
    Err(std::io::Error::other("ascension_qwen_router_metal_probe requires macOS Metal").into())
}

#[cfg(target_os = "macos")]
mod macos {
    use hawking_core::kernels::topk_softmax_batch;
    use hawking_core::metal::MetalContext;
    use hawking_core::model::qwen_moe::{
        dispatch_qwen_router_component, qwen30_route_topology_from_hf_config,
        qwen80_route_topology_from_metadata, QwenMoERouteTopology,
    };
    use serde::Serialize;
    use serde_json::{json, Value};
    use sha2::{Digest, Sha256};
    use std::env;
    use std::error::Error;
    use std::fs;
    use std::path::{Path, PathBuf};
    use std::time::Instant;

    const SCHEMA: &str = "hawking.ascension.qwen_router_metal_component_probe.v1";
    const WARMUP_RUNS: usize = 3;
    const MEASURED_RUNS: usize = 24;

    struct Args {
        qwen30_config: PathBuf,
        qwen80_metadata: PathBuf,
        out: PathBuf,
    }

    #[derive(Serialize)]
    struct TimingReport {
        warmup_runs: usize,
        measured_runs: usize,
        host_wall_us_min: f64,
        host_wall_us_p50: f64,
        host_wall_us_p95: f64,
        host_wall_us_max: f64,
        component_dispatches_per_second_from_host_wall: f64,
    }

    #[derive(Serialize)]
    struct LaneReport {
        model_id: &'static str,
        architecture: &'static str,
        n_experts: usize,
        top_k: usize,
        source_configuration_path: String,
        source_configuration_sha256: String,
        exact_topology_bound: bool,
        metal_direct_dispatch: bool,
        dispatch_count_per_router_invocation: usize,
        cpu_oracle_route_ids: Vec<u32>,
        metal_route_ids: Vec<u32>,
        ids_exact_match: bool,
        max_abs_weight_error: f32,
        weights_within_1e_minus_5: bool,
        timing: TimingReport,
    }

    fn parse_args() -> Result<Args, Box<dyn Error>> {
        let mut qwen30_config = None;
        let mut qwen80_metadata = None;
        let mut out = None;
        let mut values = env::args().skip(1);
        while let Some(flag) = values.next() {
            let value = values
                .next()
                .ok_or_else(|| format!("missing value after {flag}"))?;
            match flag.as_str() {
                "--qwen30-config" => qwen30_config = Some(PathBuf::from(value)),
                "--qwen80-metadata" => qwen80_metadata = Some(PathBuf::from(value)),
                "--out" => out = Some(PathBuf::from(value)),
                _ => return Err(format!("unknown argument {flag}").into()),
            }
        }
        Ok(Args {
            qwen30_config: qwen30_config.ok_or("--qwen30-config is required")?,
            qwen80_metadata: qwen80_metadata.ok_or("--qwen80-metadata is required")?,
            out: out.ok_or("--out is required")?,
        })
    }

    fn read_json(path: &Path) -> Result<(Value, String), Box<dyn Error>> {
        let bytes = fs::read(path)?;
        let mut hasher = Sha256::new();
        hasher.update(&bytes);
        let sha256 = format!("{:x}", hasher.finalize());
        Ok((serde_json::from_slice(&bytes)?, sha256))
    }

    fn deterministic_logits(n_experts: usize) -> Vec<f32> {
        // 37 is coprime to both supported expert counts, so this creates
        // distinct, deterministic values without route ties.
        (0..n_experts)
            .map(|expert| ((expert * 37 % n_experts) as f32 / n_experts as f32) * 12.0 - 6.0)
            .collect()
    }

    fn percentile_us(mut samples_us: Vec<f64>, percentile: f64) -> f64 {
        samples_us.sort_by(|a, b| a.total_cmp(b));
        let index = ((samples_us.len() - 1) as f64 * percentile).ceil() as usize;
        samples_us[index]
    }

    fn run_lane(
        metal: &MetalContext,
        topology: QwenMoERouteTopology,
        configuration_path: &Path,
        configuration_sha256: String,
    ) -> Result<LaneReport, Box<dyn Error>> {
        let logits = deterministic_logits(topology.n_experts);
        let mut expected_ids = vec![0u32; topology.top_k];
        let mut expected_weights = vec![0.0f32; topology.top_k];
        topk_softmax_batch(
            &logits,
            1,
            topology.n_experts,
            topology.top_k,
            &mut expected_ids,
            &mut expected_weights,
        );

        let (metal_ids, metal_weights) = dispatch_qwen_router_component(metal, topology, &logits)?;
        let ids_exact_match = expected_ids == metal_ids;
        let max_abs_weight_error = expected_weights
            .iter()
            .zip(&metal_weights)
            .map(|(expected, observed)| (expected - observed).abs())
            .fold(0.0f32, f32::max);
        let weights_within_1e_minus_5 = max_abs_weight_error <= 1.0e-5;
        if !ids_exact_match || !weights_within_1e_minus_5 {
            return Err(format!(
                "{} Metal route output failed CPU oracle: ids_match={}, max_weight_error={}",
                topology.model_id, ids_exact_match, max_abs_weight_error
            )
            .into());
        }

        for _ in 0..WARMUP_RUNS {
            let _ = dispatch_qwen_router_component(metal, topology, &logits)?;
        }
        let mut samples_us = Vec::with_capacity(MEASURED_RUNS);
        for _ in 0..MEASURED_RUNS {
            let started = Instant::now();
            let _ = dispatch_qwen_router_component(metal, topology, &logits)?;
            samples_us.push(started.elapsed().as_secs_f64() * 1_000_000.0);
        }
        let min = samples_us.iter().copied().fold(f64::INFINITY, f64::min);
        let max = samples_us.iter().copied().fold(0.0, f64::max);
        let p50 = percentile_us(samples_us.clone(), 0.50);
        let p95 = percentile_us(samples_us.clone(), 0.95);

        Ok(LaneReport {
            model_id: topology.model_id,
            architecture: topology.architecture,
            n_experts: topology.n_experts,
            top_k: topology.top_k,
            source_configuration_path: configuration_path.display().to_string(),
            source_configuration_sha256: configuration_sha256,
            exact_topology_bound: true,
            metal_direct_dispatch: true,
            dispatch_count_per_router_invocation: 1,
            cpu_oracle_route_ids: expected_ids,
            metal_route_ids: metal_ids,
            ids_exact_match,
            max_abs_weight_error,
            weights_within_1e_minus_5,
            timing: TimingReport {
                warmup_runs: WARMUP_RUNS,
                measured_runs: MEASURED_RUNS,
                host_wall_us_min: min,
                host_wall_us_p50: p50,
                host_wall_us_p95: p95,
                host_wall_us_max: max,
                component_dispatches_per_second_from_host_wall: 1_000_000.0 / p50,
            },
        })
    }

    pub fn run() -> Result<(), Box<dyn Error>> {
        let args = parse_args()?;
        let (qwen30_config, qwen30_sha256) = read_json(&args.qwen30_config)?;
        let (qwen80_metadata, qwen80_sha256) = read_json(&args.qwen80_metadata)?;
        let qwen30_topology = qwen30_route_topology_from_hf_config(&qwen30_config)?;
        let qwen80_topology = qwen80_route_topology_from_metadata(&qwen80_metadata)?;
        let metal = MetalContext::new()?;
        let qwen30 = run_lane(&metal, qwen30_topology, &args.qwen30_config, qwen30_sha256)?;
        let qwen80 = run_lane(
            &metal,
            qwen80_topology,
            &args.qwen80_metadata,
            qwen80_sha256,
        )?;
        let report = json!({
            "schema": SCHEMA,
            "status": "PASS_DIRECT_METAL_ROUTER_COMPONENT_NOT_FULL_MODEL_NOT_TPS_GATE",
            "device": metal.device_name(),
            "lanes": [qwen30, qwen80],
            "claim_boundary": {
                "direct_metal_router_component_only": true,
                "qwen30_weights_not_loaded": true,
                "qwen80_body_not_downloaded_or_loaded": true,
                "not_a_complete_decoder_or_generation_path": true,
                "not_a_gravity_artifact": true,
                "not_a_100_tps_kernel_operational_receipt": true,
                "not_a_tg3_333_tps_receipt": true,
                "not_a_manager_or_tournament_qualification": true
            }
        });
        if let Some(parent) = args.out.parent() {
            fs::create_dir_all(parent)?;
        }
        fs::write(
            &args.out,
            format!("{}\n", serde_json::to_string_pretty(&report)?),
        )?;
        println!("{}", serde_json::to_string(&report)?);
        Ok(())
    }
}

#[cfg(target_os = "macos")]
fn main() -> Result<(), Box<dyn std::error::Error>> {
    macos::run()
}

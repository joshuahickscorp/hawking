//! Bounded native-BF16 GEMV accuracy/latency comparison.
//!
//! This is an isolated, explicit benchmark. It does not inspect a sealed
//! artifact, invoke MOP, or alter the existing sequential runtime path.

#[cfg(not(target_os = "macos"))]
fn main() {
    eprintln!("gravity_native_bf16_accumulation requires macOS + Metal");
    std::process::exit(2);
}

#[cfg(target_os = "macos")]
mod macos {
    use std::error::Error;
    use std::io::Write;
    use std::path::{Path, PathBuf};
    use std::time::Instant;

    use hawking_core::gravity::{matvec_bf16_host_accumulation, NativeBf16Accumulation};
    use hawking_core::metal::{MetalContext, TokenCommandBuffer};
    use hawking_core::numeric_parity::{matvec_bf16_f64_authority, score_pair, Bounds};
    use metal::Buffer;
    use serde::Serialize;
    use serde_json::json;

    const CAMPAIGN_BASE: &str = "369fe306e9679137790d3b3614a91d51639fa658";
    const FLAGSHIP_ROWS: usize = 154_880;
    const FLAGSHIP_COLS: usize = 6_144;
    const TG: u32 = 256;

    #[derive(Debug)]
    struct Config {
        warmup: usize,
        iterations: usize,
        rounds: usize,
        max_dispatches: usize,
        max_resident_bytes: usize,
        allow_flagship_allocation: bool,
        out: PathBuf,
    }

    impl Default for Config {
        fn default() -> Self {
            Self {
                warmup: 1,
                iterations: 5,
                rounds: 5,
                max_dispatches: 200,
                max_resident_bytes: 2_000_000_000,
                allow_flagship_allocation: false,
                out: PathBuf::from("reports/base_runtime/gravity_native_bf16_accumulation.json"),
            }
        }
    }

    impl Config {
        fn parity_dispatches(&self) -> usize {
            3 * 9 * NativeBf16Accumulation::ALL.len()
        }

        fn benchmark_dispatches(&self) -> usize {
            self.rounds * (self.warmup + self.iterations) * NativeBf16Accumulation::ALL.len()
        }

        fn total_dispatches(&self) -> usize {
            self.parity_dispatches() + self.benchmark_dispatches()
        }
    }

    fn usage() -> &'static str {
        "gravity_native_bf16_accumulation \
         --allow-flagship-allocation \
         [--warmup N] [--iters N] [--rounds N] \
         [--max-dispatches N] [--max-resident-bytes N] [--out PATH]"
    }

    fn parse_usize(flag: &str, value: Option<String>) -> Result<usize, String> {
        value
            .ok_or_else(|| format!("{flag} requires a value"))?
            .parse::<usize>()
            .map_err(|e| format!("invalid {flag}: {e}"))
    }

    fn parse_config() -> Result<Config, String> {
        let mut cfg = Config::default();
        let mut args = std::env::args().skip(1);
        while let Some(arg) = args.next() {
            match arg.as_str() {
                "--warmup" => cfg.warmup = parse_usize("--warmup", args.next())?,
                "--iters" => cfg.iterations = parse_usize("--iters", args.next())?,
                "--rounds" => cfg.rounds = parse_usize("--rounds", args.next())?,
                "--max-dispatches" => {
                    cfg.max_dispatches = parse_usize("--max-dispatches", args.next())?
                }
                "--max-resident-bytes" => {
                    cfg.max_resident_bytes = parse_usize("--max-resident-bytes", args.next())?
                }
                "--allow-flagship-allocation" => cfg.allow_flagship_allocation = true,
                "--out" => {
                    cfg.out = PathBuf::from(
                        args.next()
                            .ok_or_else(|| "--out requires a path".to_string())?,
                    )
                }
                "-h" | "--help" => {
                    println!("{}", usage());
                    std::process::exit(0);
                }
                _ => return Err(format!("unknown argument {arg:?}\n{}", usage())),
            }
        }
        if !cfg.allow_flagship_allocation {
            return Err(format!(
                "exact [154880,6144] allocation requires \
                 --allow-flagship-allocation\n{}",
                usage()
            ));
        }
        if cfg.iterations == 0 || cfg.rounds == 0 {
            return Err("--iters and --rounds must be at least 1".into());
        }
        if resident_bytes(FLAGSHIP_ROWS, FLAGSHIP_COLS) > cfg.max_resident_bytes {
            return Err(format!(
                "resident allocation {} B exceeds explicit bound {} B",
                resident_bytes(FLAGSHIP_ROWS, FLAGSHIP_COLS),
                cfg.max_resident_bytes
            ));
        }
        if cfg.total_dispatches() > cfg.max_dispatches {
            return Err(format!(
                "planned {} dispatches exceeds explicit bound {}",
                cfg.total_dispatches(),
                cfg.max_dispatches
            ));
        }
        Ok(cfg)
    }

    fn resident_bytes(rows: usize, cols: usize) -> usize {
        rows * cols * 2 + cols * 4 + rows * 4
    }

    fn make_bf16_matrix_unit_scale(rows: usize, cols: usize, salt: u32) -> Vec<u8> {
        let mut bits = Vec::with_capacity(rows * cols * 2);
        for i in 0..(rows * cols) {
            let h = (i as u32).wrapping_mul(37).wrapping_add(salt);
            let sign = ((h >> 15) & 1) as u16;
            let exp = (120 + (h % 15)) as u16;
            let mant = (h & 0x7f) as u16;
            let value = (sign << 15) | (exp << 7) | mant;
            bits.extend_from_slice(&value.to_le_bytes());
        }
        bits
    }

    fn activations(cols: usize) -> Vec<(&'static str, Vec<f32>)> {
        vec![
            ("ramp", (0..cols).map(|c| (c as f32) * 0.01 - 0.3).collect()),
            (
                "mod17",
                (0..cols)
                    .map(|c| ((c * 5 + 3) % 17) as f32 * 0.05 - 0.4)
                    .collect(),
            ),
            ("ones", vec![1.0; cols]),
            ("zeros", vec![0.0; cols]),
            (
                "block3",
                (0..cols)
                    .map(|c| if c % 3 == 0 { 0.5 } else { -0.25 })
                    .collect(),
            ),
            ("sin", (0..cols).map(|c| ((c as f32).sin()) * 0.1).collect()),
            (
                "half",
                (0..cols)
                    .map(|c| if c < cols / 2 { 0.125 } else { -0.0625 })
                    .collect(),
            ),
            (
                "mod31",
                (0..cols)
                    .map(|c| ((c * 13 + 7) % 31) as f32 * 0.02 - 0.3)
                    .collect(),
            ),
            (
                "gaussish",
                (0..cols)
                    .map(|c| {
                        let u = ((c * 17 + 5) % 100) as f32 / 100.0 - 0.5;
                        (u * 3.0).tanh()
                    })
                    .collect(),
            ),
        ]
    }

    #[derive(Debug, Serialize)]
    struct ParityFailure {
        shape: [usize; 2],
        vector: &'static str,
        host_failures: Vec<String>,
        device_failures: Vec<String>,
    }

    #[derive(Debug, Serialize)]
    struct ParitySummary {
        accumulation: &'static str,
        cases: usize,
        passing_cases: usize,
        exact_host_device_cases: usize,
        decision_exact_cases: usize,
        worst_relative_l2: f64,
        worst_meaningful_relative: f64,
        worst_ulp: f64,
        failures: Vec<ParityFailure>,
    }

    impl ParitySummary {
        fn new(accumulation: NativeBf16Accumulation) -> Self {
            Self {
                accumulation: accumulation.as_str(),
                cases: 0,
                passing_cases: 0,
                exact_host_device_cases: 0,
                decision_exact_cases: 0,
                worst_relative_l2: 0.0,
                worst_meaningful_relative: 0.0,
                worst_ulp: 0.0,
                failures: Vec::new(),
            }
        }

        fn all_pass(&self) -> bool {
            self.cases == self.passing_cases
                && self.cases == self.exact_host_device_cases
                && self.cases == self.decision_exact_cases
        }
    }

    fn parity_sweep(ctx: &MetalContext) -> Result<Vec<ParitySummary>, Box<dyn Error>> {
        let bounds = Bounds::logits();
        let shapes = [(64usize, 32usize), (257usize, 17usize), (16usize, 64usize)];
        let mut summaries: Vec<_> = NativeBf16Accumulation::ALL
            .iter()
            .copied()
            .map(ParitySummary::new)
            .collect();

        for &(rows, cols) in &shapes {
            let weight = make_bf16_matrix_unit_scale(rows, cols, (rows * cols) as u32);
            let weight_buffer = ctx.new_buffer_with_bytes_checked(&weight)?;
            for (name, x) in activations(cols) {
                let authority = matvec_bf16_f64_authority(&weight, cols, &x)?;
                for (summary, accumulation) in summaries.iter_mut().zip(NativeBf16Accumulation::ALL)
                {
                    let host = matvec_bf16_host_accumulation(&weight, cols, &x, accumulation)?;
                    let device =
                        hawking_core::gravity_glm::gpu::dispatch_gemv_native_bf16_accumulation(
                            ctx,
                            &weight_buffer,
                            rows as u32,
                            cols as u32,
                            &x,
                            accumulation,
                        )?;
                    let paired = score_pair(&host, &device, &authority, &bounds);
                    summary.cases += 1;
                    if paired.pass {
                        summary.passing_cases += 1;
                    } else {
                        summary.failures.push(ParityFailure {
                            shape: [rows, cols],
                            vector: name,
                            host_failures: paired.host.failures.clone(),
                            device_failures: paired.device.failures.clone(),
                        });
                    }
                    if host == device {
                        summary.exact_host_device_cases += 1;
                    }
                    if paired.host.discrete.greedy_match
                        && paired.host.discrete.top_k_exact_match
                        && paired.device.discrete.greedy_match
                        && paired.device.discrete.top_k_exact_match
                    {
                        summary.decision_exact_cases += 1;
                    }
                    summary.worst_relative_l2 = summary
                        .worst_relative_l2
                        .max(paired.device.continuous.relative_l2);
                    summary.worst_meaningful_relative = summary
                        .worst_meaningful_relative
                        .max(paired.device.continuous.max_meaningful_rel);
                    summary.worst_ulp = summary.worst_ulp.max(paired.device.continuous.ulp.max);
                }
            }
        }
        Ok(summaries)
    }

    struct BenchFixture {
        weight: Buffer,
        x: Buffer,
        y: Buffer,
    }

    fn make_bench_fixture(ctx: &MetalContext) -> Result<BenchFixture, Box<dyn Error>> {
        let weight_bytes = FLAGSHIP_ROWS * FLAGSHIP_COLS * 2;
        let weight = ctx.new_buffer_checked(weight_bytes)?;
        // One repeated, finite normal BF16 pattern. This touches every page
        // without carrying a second 1.903 GB host allocation.
        unsafe {
            std::ptr::write_bytes(weight.contents() as *mut u8, 0x3f, weight_bytes);
        }
        let x: Vec<f32> = (0..FLAGSHIP_COLS)
            .map(|col| (col as f32 * 0.017).sin() * 0.5)
            .collect();
        Ok(BenchFixture {
            weight,
            x: ctx.new_buffer_with_bytes_checked(bytemuck::cast_slice(&x))?,
            y: ctx.new_buffer_checked(FLAGSHIP_ROWS * std::mem::size_of::<f32>())?,
        })
    }

    struct Sample {
        wall_us: f64,
        checksum: f32,
    }

    fn run_once(
        ctx: &MetalContext,
        fixture: &BenchFixture,
        accumulation: NativeBf16Accumulation,
    ) -> Result<Sample, Box<dyn Error>> {
        let rows = FLAGSHIP_ROWS as u32;
        let cols = FLAGSHIP_COLS as u32;
        let started = Instant::now();
        let mut tcb = TokenCommandBuffer::new(ctx);
        tcb.dispatch_threads(
            accumulation.metal_kernel(),
            (rows.div_ceil(TG) * TG, 1, 1),
            (TG, 1, 1),
            |encoder| {
                encoder.set_buffer(0, Some(&fixture.weight), 0);
                encoder.set_buffer(1, Some(&fixture.x), 0);
                encoder.set_buffer(2, Some(&fixture.y), 0);
                encoder.set_bytes(3, 4, &rows as *const u32 as *const _);
                encoder.set_bytes(4, 4, &cols as *const u32 as *const _);
            },
        )?;
        if tcb.dispatch_count() != 1 {
            return Err("native-BF16 benchmark expected exactly one dispatch".into());
        }
        tcb.commit_and_wait()?;
        let wall_us = started.elapsed().as_secs_f64() * 1e6;
        let out = fixture.y.contents() as *const f32;
        let checksum = unsafe { *out + *out.add(FLAGSHIP_ROWS / 2) + *out.add(FLAGSHIP_ROWS - 1) };
        Ok(Sample { wall_us, checksum })
    }

    #[derive(Debug, Clone, Serialize)]
    struct TimingSummary {
        count: usize,
        min_us: f64,
        p25_us: f64,
        median_us: f64,
        p75_us: f64,
        p95_us: f64,
        max_us: f64,
        mean_us: f64,
        stddev_us: f64,
        mad_us: f64,
        coefficient_of_variation: f64,
    }

    fn percentile(sorted: &[f64], q: f64) -> f64 {
        let position = q * (sorted.len() - 1) as f64;
        let lo = position.floor() as usize;
        let hi = position.ceil() as usize;
        if lo == hi {
            sorted[lo]
        } else {
            let fraction = position - lo as f64;
            sorted[lo] * (1.0 - fraction) + sorted[hi] * fraction
        }
    }

    fn summarize(values: &[f64]) -> TimingSummary {
        let mut sorted = values.to_vec();
        sorted.sort_by(f64::total_cmp);
        let mean = values.iter().sum::<f64>() / values.len() as f64;
        let variance = values
            .iter()
            .map(|value| (value - mean) * (value - mean))
            .sum::<f64>()
            / values.len() as f64;
        let median = percentile(&sorted, 0.5);
        let mut deviations: Vec<_> = values.iter().map(|value| (value - median).abs()).collect();
        deviations.sort_by(f64::total_cmp);
        TimingSummary {
            count: values.len(),
            min_us: sorted[0],
            p25_us: percentile(&sorted, 0.25),
            median_us: median,
            p75_us: percentile(&sorted, 0.75),
            p95_us: percentile(&sorted, 0.95),
            max_us: *sorted.last().expect("nonempty"),
            mean_us: mean,
            stddev_us: variance.sqrt(),
            mad_us: percentile(&deviations, 0.5),
            coefficient_of_variation: variance.sqrt() / mean,
        }
    }

    #[derive(Debug, Serialize)]
    struct VariantTiming {
        accumulation: &'static str,
        wall: TimingSummary,
        minimum_device_bytes_per_dispatch: usize,
        effective_weight_bandwidth_gbps_at_median: f64,
        median_overhead_vs_sequential: f64,
        checksum: f32,
    }

    fn write_receipt(path: &Path, text: &str) -> std::io::Result<()> {
        if let Some(parent) = path.parent() {
            std::fs::create_dir_all(parent)?;
        }
        let mut file = std::fs::File::create(path)?;
        file.write_all(text.as_bytes())?;
        file.write_all(b"\n")?;
        file.sync_all()
    }

    pub fn run() -> Result<(), Box<dyn Error>> {
        let cfg = parse_config().map_err(|error| format!("{error}\n{}", usage()))?;
        if std::env::var_os("HAWKING_TCB_TRACE").is_some() {
            return Err("unset HAWKING_TCB_TRACE for synchronized wall timing".into());
        }
        let ctx = MetalContext::new()?;
        for accumulation in NativeBf16Accumulation::ALL {
            let _ = ctx.pipeline(accumulation.metal_kernel())?;
        }

        eprintln!(
            "[native-bf16-accumulation] device={} exact_shape=[{},{}] \
             resident_bytes={} dispatches={}/{}",
            ctx.device_name(),
            FLAGSHIP_ROWS,
            FLAGSHIP_COLS,
            resident_bytes(FLAGSHIP_ROWS, FLAGSHIP_COLS),
            cfg.total_dispatches(),
            cfg.max_dispatches
        );

        let parity = parity_sweep(&ctx)?;
        for summary in &parity {
            eprintln!(
                "  parity {:<32} pass={}/{} exact_host_device={}/{} \
                 worst_rel_l2={:.3e} worst_meaningful_rel={:.3e} worst_ulp={:.0}",
                summary.accumulation,
                summary.passing_cases,
                summary.cases,
                summary.exact_host_device_cases,
                summary.cases,
                summary.worst_relative_l2,
                summary.worst_meaningful_relative,
                summary.worst_ulp
            );
        }
        let accurate = parity
            .iter()
            .find(|summary| {
                summary.accumulation == NativeBf16Accumulation::NeumaierCompensatedProduct.as_str()
            })
            .expect("accurate summary");
        if !accurate.all_pass() {
            return Err("accurate candidate failed parity; benchmark aborted fail-closed".into());
        }

        let allocation_started = Instant::now();
        let fixture = make_bench_fixture(&ctx)?;
        let allocation_ms = allocation_started.elapsed().as_secs_f64() * 1e3;
        let mut timings: Vec<Vec<f64>> = NativeBf16Accumulation::ALL
            .iter()
            .map(|_| Vec::new())
            .collect();
        let mut checksums = [0.0f32; 3];

        for round in 0..cfg.rounds {
            for _ in 0..cfg.warmup {
                for offset in 0..NativeBf16Accumulation::ALL.len() {
                    let index = (round + offset) % NativeBf16Accumulation::ALL.len();
                    let sample = run_once(&ctx, &fixture, NativeBf16Accumulation::ALL[index])?;
                    checksums[index] = sample.checksum;
                }
            }
            for iteration in 0..cfg.iterations {
                for offset in 0..NativeBf16Accumulation::ALL.len() {
                    let index = (round + iteration + offset) % NativeBf16Accumulation::ALL.len();
                    let sample = run_once(&ctx, &fixture, NativeBf16Accumulation::ALL[index])?;
                    timings[index].push(sample.wall_us);
                    checksums[index] = sample.checksum;
                }
            }
        }

        let summaries: Vec<_> = timings.iter().map(|values| summarize(values)).collect();
        let sequential_median = summaries[0].median_us;
        let weight_bytes = FLAGSHIP_ROWS * FLAGSHIP_COLS * 2;
        let timing_report: Vec<_> = NativeBf16Accumulation::ALL
            .iter()
            .copied()
            .zip(summaries)
            .zip(checksums)
            .map(|((accumulation, wall), checksum)| {
                let median_seconds = wall.median_us * 1e-6;
                VariantTiming {
                    accumulation: accumulation.as_str(),
                    effective_weight_bandwidth_gbps_at_median: weight_bytes as f64
                        / median_seconds
                        / 1e9,
                    median_overhead_vs_sequential: wall.median_us / sequential_median - 1.0,
                    wall,
                    minimum_device_bytes_per_dispatch: resident_bytes(FLAGSHIP_ROWS, FLAGSHIP_COLS),
                    checksum,
                }
            })
            .collect();
        for timing in &timing_report {
            eprintln!(
                "  timing {:<32} median={:.1}us p95={:.1}us cv={:.3} \
                 overhead={:+.1}% effective_weight_bw={:.1}GB/s",
                timing.accumulation,
                timing.wall.median_us,
                timing.wall.p95_us,
                timing.wall.coefficient_of_variation,
                timing.median_overhead_vs_sequential * 100.0,
                timing.effective_weight_bandwidth_gbps_at_median
            );
        }

        let accurate_timing = timing_report
            .iter()
            .find(|timing| {
                timing.accumulation == NativeBf16Accumulation::NeumaierCompensatedProduct.as_str()
            })
            .expect("accurate timing");
        let numerical_gain_costs_too_much = accurate_timing.median_overhead_vs_sequential > 0.10;

        let report = json!({
            "schema": "hawking.gravity.native_bf16_accumulation.v1",
            "campaign_base": CAMPAIGN_BASE,
            "benchmark_only": true,
            "runtime_default_changed": false,
            "device": ctx.device_name(),
            "isolation": {
                "synthetic_data_only": true,
                "sealed_artifact_used": false,
                "mop_invoked": false
            },
            "parity": {
                "policy": "unchanged Numeric Parity V2.1 Bounds::logits; exact greedy/top-5",
                "shapes": [[64, 32], [257, 17], [16, 64]],
                "vectors_per_shape": 9,
                "summaries": parity,
                "accurate_candidate_gate_passed": accurate.all_pass()
            },
            "benchmark": {
                "shape": [FLAGSHIP_ROWS, FLAGSHIP_COLS],
                "exact_flagship_geometry": true,
                "scaling_used": false,
                "allocation_bytes": resident_bytes(FLAGSHIP_ROWS, FLAGSHIP_COLS),
                "allocation_and_page_touch_ms": allocation_ms,
                "weight_bytes_per_dispatch": weight_bytes,
                "activation_bytes": FLAGSHIP_COLS * 4,
                "output_bytes": FLAGSHIP_ROWS * 4,
                "timing_source": "synchronized host wall around one command buffer / one dispatch / one wait",
                "warmup_per_variant_per_round": cfg.warmup,
                "iterations_per_variant_per_round": cfg.iterations,
                "rounds": cfg.rounds,
                "variants": timing_report
            },
            "accounting": {
                "parity_dispatches": cfg.parity_dispatches(),
                "benchmark_dispatches": cfg.benchmark_dispatches(),
                "total_dispatches": cfg.total_dispatches(),
                "max_dispatches": cfg.max_dispatches,
                "command_buffers_per_dispatch": 1,
                "dispatches_per_command_buffer": 1,
                "synchronizing_waits_per_dispatch": 1
            },
            "verdict": {
                "accurate_candidate": NativeBf16Accumulation::NeumaierCompensatedProduct.as_str(),
                "plain_neumaier_rejected_numerically": !parity[1].all_pass(),
                "accurate_candidate_numerically_green": accurate.all_pass(),
                "maximum_acceptable_median_overhead": 0.10,
                "numerical_gain_costs_too_much": numerical_gain_costs_too_much,
                "runtime_action": "none; candidate remains additive and default-off"
            },
            "caveats": [
                "The full flagship shape is exact, but benchmark weights and activations are synthetic.",
                "Effective bandwidth is weight bytes divided by synchronized wall time, not a hardware-counter measurement.",
                "The benchmark output values are not a parity authority; parity is gated separately over all inherited vectors.",
                "A single M3 Ultra result does not establish fleet-wide performance."
            ]
        });
        write_receipt(&cfg.out, &serde_json::to_string_pretty(&report)?)?;
        println!("{}", cfg.out.display());
        Ok(())
    }

    #[cfg(test)]
    mod tests {
        use super::*;

        #[test]
        fn default_run_is_bounded() {
            let config = Config {
                allow_flagship_allocation: true,
                ..Config::default()
            };
            assert_eq!(config.parity_dispatches(), 81);
            assert_eq!(config.benchmark_dispatches(), 90);
            assert_eq!(config.total_dispatches(), 171);
            assert!(config.total_dispatches() <= config.max_dispatches);
            assert_eq!(resident_bytes(FLAGSHIP_ROWS, FLAGSHIP_COLS), 1_903_809_536);
        }

        #[test]
        fn timing_summary_reports_variability() {
            let summary = summarize(&[1.0, 2.0, 3.0, 4.0, 10.0]);
            assert_eq!(summary.median_us, 3.0);
            assert_eq!(summary.mad_us, 1.0);
            assert!(summary.stddev_us > 0.0);
        }
    }
}

#[cfg(target_os = "macos")]
fn main() -> Result<(), Box<dyn std::error::Error>> {
    macos::run()
}

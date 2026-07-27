//! Bounded, synthetic flagship-geometry Metal benchmark for GLM expert waves.
//!
//! This is intentionally an isolated benchmark, not a runtime switch. It compares
//! the resident runtime's sequential projection encoding with a candidate that:
//!
//! 1. concurrently encodes every independent gate/up projection,
//! 2. preserves the dependent SiLU dispatches,
//! 3. concurrently encodes every independent down projection, and
//! 4. preserves the ordered weighted combine.
//!
//! The payload is generated in memory at direct-u8 R4-like geometry
//! (D=32/S=1/sub=32/card=256/bits=8). No sealed artifact or MOP is read.

#[cfg(not(target_os = "macos"))]
fn main() {
    eprintln!("gravity_glm_expert_wave_concurrent requires macOS + Metal");
    std::process::exit(2);
}

#[cfg(target_os = "macos")]
mod macos {
    use std::error::Error;
    use std::io::Write;
    use std::path::{Path, PathBuf};
    use std::time::Instant;

    use half::f16;
    use hawking_core::metal::{MetalContext, TokenCommandBuffer};
    use hawking_core::numeric_parity::{score_against_f64, Bounds};
    use metal::Buffer;
    use serde::Serialize;
    use serde_json::json;

    const CAMPAIGN_BASE: &str = "b27b1e6f85138a6264484937906f371aab4299be";
    const HIDDEN: usize = 6144;
    const INTER: usize = 2048;
    const ROUTED_EXPERTS: usize = 8;
    const SHARED_EXPERTS: usize = 1;
    const EXPERTS: usize = ROUTED_EXPERTS + SHARED_EXPERTS;
    const D: usize = 32;
    const SUBSPACES: usize = 1;
    const SUB: usize = 32;
    const CARD: usize = 256;
    const BITS: usize = 8;
    const THREADGROUP: u32 = 256;
    const DISPATCHES_PER_WAVE: usize = EXPERTS * 5;

    const MATVEC_KERNEL: &str = "gravity_pq_matvec_bits8_direct";
    const SILU_KERNEL: &str = "gravity_silu_mul_f32";
    const AXPY_KERNEL: &str = "gravity_axpy_f32";

    #[derive(Debug)]
    struct Config {
        warmup: usize,
        iterations: usize,
        rounds: usize,
        max_dispatches: usize,
        allow_synthetic_flagship: bool,
        out: PathBuf,
    }

    impl Default for Config {
        fn default() -> Self {
            Self {
                warmup: 1,
                iterations: 5,
                rounds: 5,
                max_dispatches: 3_000,
                allow_synthetic_flagship: false,
                out: PathBuf::from("reports/base_runtime/gravity_glm_expert_wave_concurrent.json"),
            }
        }
    }

    impl Config {
        fn planned_waves(&self) -> usize {
            2 + self.rounds * 2 * (self.warmup + self.iterations)
        }

        fn planned_dispatches(&self) -> usize {
            self.planned_waves() * DISPATCHES_PER_WAVE
        }
    }

    fn usage() -> &'static str {
        "gravity_glm_expert_wave_concurrent \
         --allow-synthetic-flagship \
         [--warmup N] [--iters N] [--rounds N] \
         [--max-dispatches N] [--out PATH]"
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
                "--allow-synthetic-flagship" => cfg.allow_synthetic_flagship = true,
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
        if !cfg.allow_synthetic_flagship {
            return Err(format!(
                "full synthetic flagship geometry requires explicit \
                 --allow-synthetic-flagship acknowledgement\n{}",
                usage()
            ));
        }
        if cfg.iterations == 0 || cfg.rounds == 0 {
            return Err("--iters and --rounds must be at least 1".into());
        }
        if cfg.planned_dispatches() > cfg.max_dispatches {
            return Err(format!(
                "bounded benchmark rejected: planned {} Metal dispatches exceeds \
                 --max-dispatches {}; reduce warmup/iters/rounds or explicitly raise the bound",
                cfg.planned_dispatches(),
                cfg.max_dispatches
            ));
        }
        Ok(cfg)
    }

    /// Metal `GravityPQParams` mirror: eight uints in shader declaration order.
    #[repr(C)]
    #[derive(Clone, Copy)]
    struct PqParams {
        dim: u32,
        subspaces: u32,
        sub: u32,
        card: u32,
        rows: u32,
        cols: u32,
        nchunk: u32,
        bits: u32,
    }

    struct PqMatrix {
        params: PqParams,
        host_codebook: Vec<f16>,
        host_codes: Vec<u8>,
        codebook: Buffer,
        codes: Buffer,
    }

    struct Fixture {
        x_host: Vec<f32>,
        x: Buffer,
        gate: Vec<PqMatrix>,
        up: Vec<PqMatrix>,
        down: Vec<PqMatrix>,
        gate_out: Vec<Buffer>,
        up_out: Vec<Buffer>,
        act: Vec<Buffer>,
        down_out: Vec<Buffer>,
        combined: Buffer,
        scales: [f32; EXPERTS],
    }

    fn next_u32(state: &mut u64) -> u32 {
        let mut x = *state;
        x ^= x << 13;
        x ^= x >> 7;
        x ^= x << 17;
        *state = x;
        (x >> 16) as u32
    }

    fn make_matrix(
        ctx: &MetalContext,
        rows: usize,
        cols: usize,
        seed: u64,
    ) -> Result<PqMatrix, Box<dyn Error>> {
        assert_eq!(cols % D, 0);
        let mut state = seed;
        let mut host_codebook = Vec::with_capacity(SUBSPACES * CARD * SUB);
        for _ in 0..SUBSPACES * CARD * SUB {
            // Positive, modest weights keep the FP64 authority well-conditioned
            // while still varying every independently allocated matrix.
            let unit = (next_u32(&mut state) % 1_501) as f32;
            host_codebook.push(f16::from_f32(0.000_4 + unit * 0.000_001));
        }
        let mut host_codes = Vec::with_capacity(rows * (cols / D) * SUBSPACES);
        for _ in 0..host_codes.capacity() {
            host_codes.push(next_u32(&mut state) as u8);
        }
        let codebook = ctx.new_buffer_with_bytes_checked(bytemuck::cast_slice(&host_codebook))?;
        let codes = ctx.new_buffer_with_bytes_checked(&host_codes)?;
        Ok(PqMatrix {
            params: PqParams {
                dim: D as u32,
                subspaces: SUBSPACES as u32,
                sub: SUB as u32,
                card: CARD as u32,
                rows: rows as u32,
                cols: cols as u32,
                nchunk: (cols / D) as u32,
                bits: BITS as u32,
            },
            host_codebook,
            host_codes,
            codebook,
            codes,
        })
    }

    fn allocate_many(
        ctx: &MetalContext,
        count: usize,
        elements: usize,
    ) -> Result<Vec<Buffer>, Box<dyn Error>> {
        (0..count)
            .map(|_| {
                ctx.new_buffer_checked(elements * std::mem::size_of::<f32>())
                    .map_err(Into::into)
            })
            .collect()
    }

    fn make_fixture(ctx: &MetalContext) -> Result<Fixture, Box<dyn Error>> {
        let x_host: Vec<f32> = (0..HIDDEN)
            .map(|i| 0.01 + ((i * 37 + 11) % 1_001) as f32 * 0.000_01)
            .collect();
        let x = ctx.new_buffer_with_bytes_checked(bytemuck::cast_slice(&x_host))?;

        let mut gate = Vec::with_capacity(EXPERTS);
        let mut up = Vec::with_capacity(EXPERTS);
        let mut down = Vec::with_capacity(EXPERTS);
        for expert in 0..EXPERTS {
            let base = 0x94d0_49bb_1331_11ebu64 ^ (expert as u64).wrapping_mul(0x9e37_79b9);
            gate.push(make_matrix(ctx, INTER, HIDDEN, base ^ 0x1111_0001)?);
            up.push(make_matrix(ctx, INTER, HIDDEN, base ^ 0x2222_0002)?);
            down.push(make_matrix(ctx, HIDDEN, INTER, base ^ 0x3333_0003)?);
        }

        Ok(Fixture {
            x_host,
            x,
            gate,
            up,
            down,
            gate_out: allocate_many(ctx, EXPERTS, INTER)?,
            up_out: allocate_many(ctx, EXPERTS, INTER)?,
            act: allocate_many(ctx, EXPERTS, INTER)?,
            down_out: allocate_many(ctx, EXPERTS, HIDDEN)?,
            combined: ctx.new_buffer_checked(HIDDEN * std::mem::size_of::<f32>())?,
            scales: [0.13, 0.11, 0.09, 0.14, 0.08, 0.12, 0.10, 0.07, 1.0],
        })
    }

    fn cpu_matvec(matrix: &PqMatrix, x: &[f64]) -> Vec<f64> {
        let rows = matrix.params.rows as usize;
        let cols = matrix.params.cols as usize;
        let nchunk = matrix.params.nchunk as usize;
        assert_eq!(x.len(), cols);
        let mut y = vec![0.0f64; rows];
        for (row, out) in y.iter_mut().enumerate() {
            let mut acc = 0.0f64;
            for chunk in 0..nchunk {
                let code = matrix.host_codes[row * nchunk + chunk] as usize;
                let entry = &matrix.host_codebook[code * SUB..(code + 1) * SUB];
                let xs = &x[chunk * D..chunk * D + SUB];
                for (&weight, &value) in entry.iter().zip(xs) {
                    acc += f32::from(weight) as f64 * value;
                }
            }
            *out = acc;
        }
        y
    }

    fn fp64_authority(fixture: &Fixture) -> Vec<f64> {
        let x: Vec<f64> = fixture.x_host.iter().map(|&v| v as f64).collect();
        let mut combined = vec![0.0f64; HIDDEN];
        for expert in 0..EXPERTS {
            let gate = cpu_matvec(&fixture.gate[expert], &x);
            let up = cpu_matvec(&fixture.up[expert], &x);
            let act: Vec<f64> = gate
                .iter()
                .zip(&up)
                .map(|(&g, &u)| (g / (1.0 + (-g).exp())) * u)
                .collect();
            let down = cpu_matvec(&fixture.down[expert], &act);
            let scale = fixture.scales[expert] as f64;
            for (dst, value) in combined.iter_mut().zip(down) {
                *dst += scale * value;
            }
        }
        combined
    }

    fn encode_matvec(
        tcb: &mut TokenCommandBuffer<'_>,
        matrix: &PqMatrix,
        x: &Buffer,
        y: &Buffer,
    ) -> Result<(), Box<dyn Error>> {
        let params = matrix.params;
        let n_threadgroups = params.rows.div_ceil(8);
        tcb.dispatch_threads(
            MATVEC_KERNEL,
            (n_threadgroups * THREADGROUP, 1, 1),
            (THREADGROUP, 1, 1),
            |enc| {
                enc.set_buffer(0, Some(&matrix.codebook), 0);
                enc.set_buffer(1, Some(&matrix.codes), 0);
                enc.set_buffer(2, Some(x), 0);
                enc.set_buffer(3, Some(y), 0);
                enc.set_bytes(
                    4,
                    std::mem::size_of::<PqParams>() as u64,
                    &params as *const PqParams as *const _,
                );
            },
        )?;
        Ok(())
    }

    fn encode_silu(
        tcb: &mut TokenCommandBuffer<'_>,
        gate: &Buffer,
        up: &Buffer,
        out: &Buffer,
    ) -> Result<(), Box<dyn Error>> {
        let n = INTER as u32;
        tcb.dispatch_threads(
            SILU_KERNEL,
            (n.div_ceil(THREADGROUP) * THREADGROUP, 1, 1),
            (THREADGROUP, 1, 1),
            |enc| {
                enc.set_buffer(0, Some(gate), 0);
                enc.set_buffer(1, Some(up), 0);
                enc.set_buffer(2, Some(out), 0);
                enc.set_bytes(3, 4, &n as *const u32 as *const _);
            },
        )?;
        Ok(())
    }

    fn encode_axpy(
        tcb: &mut TokenCommandBuffer<'_>,
        y: &Buffer,
        x: &Buffer,
        scale: f32,
    ) -> Result<(), Box<dyn Error>> {
        let n = HIDDEN as u32;
        tcb.dispatch_threads(
            AXPY_KERNEL,
            (n.div_ceil(THREADGROUP) * THREADGROUP, 1, 1),
            (THREADGROUP, 1, 1),
            |enc| {
                enc.set_buffer(0, Some(y), 0);
                enc.set_buffer(1, Some(x), 0);
                enc.set_bytes(2, 4, &scale as *const f32 as *const _);
                enc.set_bytes(3, 4, &n as *const u32 as *const _);
            },
        )?;
        Ok(())
    }

    fn zero_f32(buffer: &Buffer, len: usize) {
        assert!(buffer.length() >= (len * std::mem::size_of::<f32>()) as u64);
        unsafe {
            std::ptr::write_bytes(buffer.contents() as *mut f32, 0, len);
        }
    }

    fn read_f32(buffer: &Buffer, len: usize) -> Vec<f32> {
        assert!(buffer.length() >= (len * std::mem::size_of::<f32>()) as u64);
        unsafe { std::slice::from_raw_parts(buffer.contents() as *const f32, len) }.to_vec()
    }

    #[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize)]
    #[serde(rename_all = "snake_case")]
    enum Strategy {
        Sequential,
        ConcurrentProjectionGroups,
    }

    impl Strategy {
        fn as_str(self) -> &'static str {
            match self {
                Self::Sequential => "sequential",
                Self::ConcurrentProjectionGroups => "concurrent_projection_groups",
            }
        }
    }

    struct WaveSample {
        output: Vec<f32>,
        wall_us: f64,
        dispatches: usize,
    }

    fn run_wave(
        ctx: &MetalContext,
        fixture: &Fixture,
        strategy: Strategy,
    ) -> Result<WaveSample, Box<dyn Error>> {
        let started = Instant::now();
        zero_f32(&fixture.combined, HIDDEN);
        let mut wave = TokenCommandBuffer::new(ctx);

        if strategy == Strategy::ConcurrentProjectionGroups {
            wave.begin_concurrent_group()?;
        }
        for expert in 0..EXPERTS {
            encode_matvec(
                &mut wave,
                &fixture.gate[expert],
                &fixture.x,
                &fixture.gate_out[expert],
            )?;
            encode_matvec(
                &mut wave,
                &fixture.up[expert],
                &fixture.x,
                &fixture.up_out[expert],
            )?;
        }
        if strategy == Strategy::ConcurrentProjectionGroups {
            wave.end_concurrent_group()?;
        }

        // These dispatches depend on the completed gate/up projection group.
        for expert in 0..EXPERTS {
            encode_silu(
                &mut wave,
                &fixture.gate_out[expert],
                &fixture.up_out[expert],
                &fixture.act[expert],
            )?;
        }

        if strategy == Strategy::ConcurrentProjectionGroups {
            wave.begin_concurrent_group()?;
        }
        for expert in 0..EXPERTS {
            encode_matvec(
                &mut wave,
                &fixture.down[expert],
                &fixture.act[expert],
                &fixture.down_out[expert],
            )?;
        }
        if strategy == Strategy::ConcurrentProjectionGroups {
            wave.end_concurrent_group()?;
        }

        // Preserve the runtime's prefix-order accumulation exactly.
        for expert in 0..EXPERTS {
            encode_axpy(
                &mut wave,
                &fixture.combined,
                &fixture.down_out[expert],
                fixture.scales[expert],
            )?;
        }

        let dispatches = wave.dispatch_count();
        if dispatches != DISPATCHES_PER_WAVE {
            return Err(format!(
                "{} encoded {dispatches} dispatches; expected {DISPATCHES_PER_WAVE}",
                strategy.as_str()
            )
            .into());
        }
        wave.commit_and_wait()?;
        let output = read_f32(&fixture.combined, HIDDEN);
        let wall_us = started.elapsed().as_secs_f64() * 1e6;
        Ok(WaveSample {
            output,
            wall_us,
            dispatches,
        })
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

    #[derive(Debug, Clone, Serialize)]
    struct RatioSummary {
        count: usize,
        min: f64,
        p25: f64,
        median: f64,
        p75: f64,
        p95: f64,
        max: f64,
        mean: f64,
        stddev: f64,
        mad: f64,
        coefficient_of_variation: f64,
    }

    impl From<&TimingSummary> for RatioSummary {
        fn from(summary: &TimingSummary) -> Self {
            Self {
                count: summary.count,
                min: summary.min_us,
                p25: summary.p25_us,
                median: summary.median_us,
                p75: summary.p75_us,
                p95: summary.p95_us,
                max: summary.max_us,
                mean: summary.mean_us,
                stddev: summary.stddev_us,
                mad: summary.mad_us,
                coefficient_of_variation: summary.coefficient_of_variation,
            }
        }
    }

    fn percentile(sorted: &[f64], q: f64) -> f64 {
        assert!(!sorted.is_empty());
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
        assert!(!values.is_empty());
        let mut sorted = values.to_vec();
        sorted.sort_by(f64::total_cmp);
        let mean = values.iter().sum::<f64>() / values.len() as f64;
        let variance = values
            .iter()
            .map(|value| {
                let delta = value - mean;
                delta * delta
            })
            .sum::<f64>()
            / values.len() as f64;
        let median = percentile(&sorted, 0.5);
        let mut absolute_deviations: Vec<f64> =
            values.iter().map(|value| (value - median).abs()).collect();
        absolute_deviations.sort_by(f64::total_cmp);
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
            mad_us: percentile(&absolute_deviations, 0.5),
            coefficient_of_variation: if mean == 0.0 {
                0.0
            } else {
                variance.sqrt() / mean
            },
        }
    }

    #[derive(Serialize)]
    struct RoundReport {
        round: usize,
        first_timed_strategy: Strategy,
        sequential: TimingSummary,
        concurrent_projection_groups: TimingSummary,
        paired_speedup: RatioSummary,
        candidate_faster_pairs: usize,
    }

    #[derive(Default)]
    struct Accounting {
        waves: usize,
        command_buffers: usize,
        waits: usize,
        dispatches: usize,
    }

    impl Accounting {
        fn record(&mut self, sample: &WaveSample) {
            self.waves += 1;
            self.command_buffers += 1;
            self.waits += 1;
            self.dispatches += sample.dispatches;
        }
    }

    fn exact_difference(left: &[f32], right: &[f32]) -> (usize, f64) {
        assert_eq!(left.len(), right.len());
        let mismatches = left
            .iter()
            .zip(right)
            .filter(|(a, b)| a.to_bits() != b.to_bits())
            .count();
        let max_abs = left
            .iter()
            .zip(right)
            .map(|(&a, &b)| (a as f64 - b as f64).abs())
            .fold(0.0, f64::max);
        (mismatches, max_abs)
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
        let cfg = parse_config().map_err(|e| format!("{e}\n{}", usage()))?;
        if std::env::var_os("HAWKING_TCB_TRACE").is_some() {
            return Err(
                "unset HAWKING_TCB_TRACE: GPU trace modes intentionally disable concurrent groups"
                    .into(),
            );
        }

        let ctx = MetalContext::new()?;
        eprintln!(
            "[expert-wave-concurrent] device={} hidden={} inter={} experts={} \
             warmup={} iters={} rounds={} bounded_dispatches={}/{}",
            ctx.device_name(),
            HIDDEN,
            INTER,
            EXPERTS,
            cfg.warmup,
            cfg.iterations,
            cfg.rounds,
            cfg.planned_dispatches(),
            cfg.max_dispatches
        );

        // Compile pipelines before parity and timing.
        let _ = ctx.pipeline(MATVEC_KERNEL)?;
        let _ = ctx.pipeline(SILU_KERNEL)?;
        let _ = ctx.pipeline(AXPY_KERNEL)?;

        let fixture_started = Instant::now();
        let fixture = make_fixture(&ctx)?;
        let fixture_build_ms = fixture_started.elapsed().as_secs_f64() * 1e3;

        let authority_started = Instant::now();
        let authority = fp64_authority(&fixture);
        let authority_ms = authority_started.elapsed().as_secs_f64() * 1e3;

        let mut accounting = Accounting::default();
        let sequential_parity = run_wave(&ctx, &fixture, Strategy::Sequential)?;
        accounting.record(&sequential_parity);
        let concurrent_parity = run_wave(&ctx, &fixture, Strategy::ConcurrentProjectionGroups)?;
        accounting.record(&concurrent_parity);

        let sequential_score = score_against_f64(
            &sequential_parity.output,
            &authority,
            &Bounds::continuous_only(),
            "sequential_direct_u8",
        );
        let concurrent_score = score_against_f64(
            &concurrent_parity.output,
            &authority,
            &Bounds::continuous_only(),
            "concurrent_projection_groups_direct_u8",
        );
        let (bit_mismatches, exact_max_abs) =
            exact_difference(&sequential_parity.output, &concurrent_parity.output);
        let exact_match = bit_mismatches == 0;
        let parity_gate_passed = exact_match && sequential_score.pass && concurrent_score.pass;
        eprintln!(
            "  parity exact={} mismatches={} max_abs={:.3e} seq_v2.1={} conc_v2.1={} \
             fp64_authority_ms={:.1}",
            exact_match,
            bit_mismatches,
            exact_max_abs,
            sequential_score.pass,
            concurrent_score.pass,
            authority_ms
        );

        let mut round_reports = Vec::with_capacity(cfg.rounds);
        let mut all_sequential = Vec::with_capacity(cfg.rounds * cfg.iterations);
        let mut all_concurrent = Vec::with_capacity(cfg.rounds * cfg.iterations);
        let mut all_paired_speedups = Vec::with_capacity(cfg.rounds * cfg.iterations);

        for round in 0..cfg.rounds {
            let warmup_order = if round % 2 == 0 {
                [Strategy::Sequential, Strategy::ConcurrentProjectionGroups]
            } else {
                [Strategy::ConcurrentProjectionGroups, Strategy::Sequential]
            };
            for _ in 0..cfg.warmup {
                for strategy in warmup_order {
                    let sample = run_wave(&ctx, &fixture, strategy)?;
                    accounting.record(&sample);
                }
            }

            let first_timed_strategy = if round % 2 == 0 {
                Strategy::Sequential
            } else {
                Strategy::ConcurrentProjectionGroups
            };
            let mut sequential_us = Vec::with_capacity(cfg.iterations);
            let mut concurrent_us = Vec::with_capacity(cfg.iterations);
            let mut paired_speedups = Vec::with_capacity(cfg.iterations);
            let mut candidate_faster_pairs = 0usize;
            for iteration in 0..cfg.iterations {
                let sequential_first = (round + iteration) % 2 == 0;
                let order = if sequential_first {
                    [Strategy::Sequential, Strategy::ConcurrentProjectionGroups]
                } else {
                    [Strategy::ConcurrentProjectionGroups, Strategy::Sequential]
                };
                let mut pair = [0.0f64; 2];
                for strategy in order {
                    let sample = run_wave(&ctx, &fixture, strategy)?;
                    accounting.record(&sample);
                    match strategy {
                        Strategy::Sequential => pair[0] = sample.wall_us,
                        Strategy::ConcurrentProjectionGroups => pair[1] = sample.wall_us,
                    }
                }
                sequential_us.push(pair[0]);
                concurrent_us.push(pair[1]);
                let speedup = pair[0] / pair[1];
                paired_speedups.push(speedup);
                if speedup > 1.0 {
                    candidate_faster_pairs += 1;
                }
            }

            let sequential_summary = summarize(&sequential_us);
            let concurrent_summary = summarize(&concurrent_us);
            let paired_summary = summarize(&paired_speedups);
            eprintln!(
                "  round={} seq_med={:.1}us conc_med={:.1}us median_paired_speedup={:.4}x \
                 faster_pairs={}/{}",
                round + 1,
                sequential_summary.median_us,
                concurrent_summary.median_us,
                paired_summary.median_us,
                candidate_faster_pairs,
                cfg.iterations
            );
            all_sequential.extend_from_slice(&sequential_us);
            all_concurrent.extend_from_slice(&concurrent_us);
            all_paired_speedups.extend_from_slice(&paired_speedups);
            round_reports.push(RoundReport {
                round: round + 1,
                first_timed_strategy,
                sequential: sequential_summary,
                concurrent_projection_groups: concurrent_summary,
                paired_speedup: RatioSummary::from(&paired_summary),
                candidate_faster_pairs,
            });
        }

        let sequential_summary = summarize(&all_sequential);
        let concurrent_summary = summarize(&all_concurrent);
        let paired_speedup_summary = summarize(&all_paired_speedups);
        let faster_rounds = round_reports
            .iter()
            .filter(|round| {
                round.concurrent_projection_groups.median_us < round.sequential.median_us
            })
            .count();
        let faster_pairs = all_paired_speedups
            .iter()
            .filter(|&&speedup| speedup > 1.0)
            .count();
        let required_faster_pairs = (all_paired_speedups.len() * 3).div_ceil(4);
        let candidate_improvement_replicated = parity_gate_passed
            && faster_rounds == cfg.rounds
            && faster_pairs >= required_faster_pairs
            && paired_speedup_summary.median_us >= 1.03;

        if accounting.waves != cfg.planned_waves()
            || accounting.dispatches != cfg.planned_dispatches()
            || accounting.command_buffers != cfg.planned_waves()
            || accounting.waits != cfg.planned_waves()
        {
            return Err(format!(
                "accounting mismatch: waves={} CBs={} waits={} dispatches={}; planned={}/{}/{}/{}",
                accounting.waves,
                accounting.command_buffers,
                accounting.waits,
                accounting.dispatches,
                cfg.planned_waves(),
                cfg.planned_waves(),
                cfg.planned_waves(),
                cfg.planned_dispatches()
            )
            .into());
        }

        let report = json!({
            "schema": "hawking.gravity_glm.expert_wave_concurrent.v1",
            "campaign_base": CAMPAIGN_BASE,
            "benchmark_only": true,
            "runtime_wired_or_promoted": false,
            "device": ctx.device_name(),
            "timing_source": {
                "kind": "synchronized_host_wall",
                "scope": "host zero + command-buffer allocation/encode/commit/wait + shared-buffer output copy",
                "gpu_timestamps": false,
                "reason": "HAWKING_TCB_TRACE GPU modes intentionally disable concurrent groups"
            },
            "payload": {
                "source": "deterministic in-memory synthetic data",
                "sealed_artifact_used": false,
                "mop_invoked": false,
                "geometry": {
                    "hidden": HIDDEN,
                    "intermediate": INTER,
                    "routed_experts": ROUTED_EXPERTS,
                    "shared_experts": SHARED_EXPERTS,
                    "total_experts": EXPERTS,
                    "pq": {
                        "encoding": "direct_u8",
                        "d": D,
                        "subspaces": SUBSPACES,
                        "sub": SUB,
                        "cardinality": CARD,
                        "bits": BITS
                    },
                    "projection_matrices": EXPERTS * 3
                },
                "fixture_build_ms": fixture_build_ms
            },
            "schedules": {
                "sequential": [
                    {"stage": "gate_up", "dispatches": EXPERTS * 2, "encoder": "ordered per-dispatch"},
                    {"stage": "silu", "dispatches": EXPERTS, "encoder": "ordered per-dispatch"},
                    {"stage": "down", "dispatches": EXPERTS, "encoder": "ordered per-dispatch"},
                    {"stage": "weighted_combine", "dispatches": EXPERTS, "encoder": "ordered per-dispatch"}
                ],
                "concurrent_projection_groups": [
                    {"stage": "all independent gate/up projections", "dispatches": EXPERTS * 2, "encoder": "one MTLDispatchTypeConcurrent group"},
                    {"stage": "dependent silu", "dispatches": EXPERTS, "encoder": "ordered per-dispatch"},
                    {"stage": "all independent down projections", "dispatches": EXPERTS, "encoder": "one MTLDispatchTypeConcurrent group"},
                    {"stage": "ordered weighted combine", "dispatches": EXPERTS, "encoder": "ordered per-dispatch"}
                ]
            },
            "per_wave_accounting": {
                "command_buffers": 1,
                "synchronizing_waits": 1,
                "dispatches": DISPATCHES_PER_WAVE,
                "gate_up_dispatches": EXPERTS * 2,
                "silu_dispatches": EXPERTS,
                "down_dispatches": EXPERTS,
                "ordered_combine_dispatches": EXPERTS,
                "candidate_concurrent_groups": 2
            },
            "run_accounting": {
                "warmup_per_strategy_per_round": cfg.warmup,
                "iterations_per_strategy_per_round": cfg.iterations,
                "rounds": cfg.rounds,
                "waves": accounting.waves,
                "command_buffers": accounting.command_buffers,
                "synchronizing_waits": accounting.waits,
                "dispatches": accounting.dispatches,
                "max_dispatches": cfg.max_dispatches
            },
            "parity": {
                "authority": "FP64 decode-and-wave authority over the same synthetic direct-u8 payload",
                "authority_compute_ms": authority_ms,
                "exact_sequential_vs_concurrent": {
                    "pass": exact_match,
                    "bit_mismatches": bit_mismatches,
                    "max_abs_difference": exact_max_abs
                },
                "numeric_parity_v2_1": {
                    "bounds": Bounds::continuous_only(),
                    "sequential": sequential_score,
                    "concurrent_projection_groups": concurrent_score
                },
                "gate_passed": parity_gate_passed
            },
            "timing": {
                "rounds": round_reports,
                "aggregate": {
                    "sequential": sequential_summary,
                    "concurrent_projection_groups": concurrent_summary,
                    "paired_speedup_sequential_over_concurrent": RatioSummary::from(&paired_speedup_summary),
                    "candidate_faster_rounds": faster_rounds,
                    "candidate_faster_pairs": faster_pairs,
                    "required_faster_pairs": required_faster_pairs
                }
            },
            "promotion_rule": {
                "definition": "parity green; candidate median wins every round; >=75% paired wins; median paired speedup >=1.03x",
                "candidate_improvement_replicated": candidate_improvement_replicated,
                "runtime_action": "none; this change contains no runtime wiring even if the isolated gate passes"
            },
            "caveats": [
                "Synthetic weights and activations reproduce flagship shapes and direct-u8 memory geometry, not a sealed model's value distribution.",
                "Synchronized host wall time includes encoding and host work; per-dispatch GPU timestamps cannot be collected without disabling concurrent groups.",
                "One isolated device result is evidence for or against this scheduling idea, not a fleet-wide promotion decision.",
                "Concurrent encoder correctness relies on disjoint output buffers inside each group; dependency boundaries and ordered accumulation are deliberately retained."
            ]
        });

        let text = serde_json::to_string_pretty(&report)?;
        write_receipt(&cfg.out, &text)?;
        println!("{}", cfg.out.display());

        if !parity_gate_passed {
            return Err(format!(
                "parity gate failed; negative receipt written to {}",
                cfg.out.display()
            )
            .into());
        }
        Ok(())
    }

    #[cfg(test)]
    mod tests {
        use super::*;

        #[test]
        fn default_run_is_bounded_and_accounted() {
            let cfg = Config {
                allow_synthetic_flagship: true,
                ..Config::default()
            };
            assert_eq!(DISPATCHES_PER_WAVE, 45);
            assert_eq!(cfg.planned_waves(), 62);
            assert_eq!(cfg.planned_dispatches(), 2_790);
            assert!(cfg.planned_dispatches() <= cfg.max_dispatches);
        }

        #[test]
        fn summary_reports_variability_and_interpolated_percentiles() {
            let summary = summarize(&[10.0, 20.0, 30.0, 40.0, 100.0]);
            assert_eq!(summary.count, 5);
            assert_eq!(summary.median_us, 30.0);
            assert_eq!(summary.p25_us, 20.0);
            assert_eq!(summary.p75_us, 40.0);
            assert!(summary.stddev_us > 0.0);
            assert_eq!(summary.mad_us, 10.0);
        }
    }
}

#[cfg(target_os = "macos")]
fn main() -> Result<(), Box<dyn std::error::Error>> {
    macos::run()
}

//! Bounded real-geometry autotune harness for additive gravity-pq kernels.
//!
//! The synthetic payload uses the sealed-artifact census geometry
//! D=32/S=1/sub=32/card=256/bits=8 and the two dominant routed shapes. It
//! allocates full row/code streams, keeps them resident across candidates,
//! gates every candidate against an FP64 authority under Numeric Parity V2.1,
//! and only ranks parity-green candidates. It never changes the runtime
//! default. Full census geometries additionally require the explicit
//! `--allow-exact-geometry` acknowledgement; this harness is never an
//! automatic test.

#[cfg(not(target_os = "macos"))]
fn main() {
    eprintln!("gravity_pq_autotune requires macOS + Metal");
    std::process::exit(2);
}

#[cfg(target_os = "macos")]
mod macos {
    use std::io::Write;
    use std::path::PathBuf;
    use std::str::FromStr;

    use half::f16;
    use hawking_core::gravity::{
        pq_matvec_f64_authority, PqMetalBenchmark, PqMetalKernelVariant, PqMetalMatrix,
        PqMetalTimingSummary,
    };
    use hawking_core::metal::MetalContext;
    use hawking_core::numeric_parity::{score_against_f64, Bounds};
    use serde::Serialize;
    use serde_json::{json, Value};

    const D: u16 = 32;
    const SUBSPACES: u16 = 1;
    const SUB: u16 = 32;
    const CARD: u16 = 256;
    const BITS: u16 = 8;

    struct Config {
        geometries: Vec<(u32, u32)>,
        variants: Vec<PqMetalKernelVariant>,
        warmup: usize,
        iterations: usize,
        rounds: usize,
        max_dispatches: usize,
        allow_exact_geometry: bool,
        out: PathBuf,
    }

    impl Default for Config {
        fn default() -> Self {
            Self {
                geometries: vec![(2048, 6144), (6144, 2048)],
                variants: PqMetalKernelVariant::ALL.to_vec(),
                warmup: 2,
                // Six candidates total eight dispatches per pass. Eleven
                // samples keep the acknowledged full sweep at exactly 640.
                iterations: 11,
                rounds: 3,
                max_dispatches: 640,
                allow_exact_geometry: false,
                out: PathBuf::from("reports/base_runtime/gravity_pq_autotune.json"),
            }
        }
    }

    fn usage() -> &'static str {
        "gravity_pq_autotune \
         [--geometry ROWSxCOLS ...] \
         [--variants generic,bits8-direct,bits8-vec4,bits8-double-single,bits8-2d-split4,bits8-2d-split8] \
         [--allow-exact-geometry] \
         [--warmup N] [--iters N] [--rounds N] [--max-dispatches N] [--out PATH]"
    }

    fn parse_usize(flag: &str, value: Option<String>) -> Result<usize, String> {
        value
            .ok_or_else(|| format!("{flag} requires a value"))?
            .parse::<usize>()
            .map_err(|e| format!("invalid {flag}: {e}"))
    }

    fn parse_geometry(value: &str) -> Result<(u32, u32), String> {
        let (rows, cols) = value
            .split_once('x')
            .or_else(|| value.split_once('X'))
            .ok_or_else(|| format!("invalid geometry {value:?}; expected ROWSxCOLS"))?;
        let rows = rows
            .parse::<u32>()
            .map_err(|e| format!("invalid rows in {value:?}: {e}"))?;
        let cols = cols
            .parse::<u32>()
            .map_err(|e| format!("invalid cols in {value:?}: {e}"))?;
        if rows == 0 || cols == 0 || cols % D as u32 != 0 {
            return Err(format!(
                "geometry {value:?} must be nonzero and cols must be divisible by D={D}"
            ));
        }
        Ok((rows, cols))
    }

    fn parse_config() -> Result<Config, String> {
        let mut cfg = Config::default();
        let mut explicit_geometries = Vec::new();
        let mut args = std::env::args().skip(1);
        while let Some(arg) = args.next() {
            match arg.as_str() {
                "--geometry" => {
                    let raw = args
                        .next()
                        .ok_or_else(|| "--geometry requires ROWSxCOLS".to_string())?;
                    explicit_geometries.push(parse_geometry(&raw)?);
                }
                "--variants" => {
                    let raw = args
                        .next()
                        .ok_or_else(|| "--variants requires a comma-separated list".to_string())?;
                    cfg.variants = raw
                        .split(',')
                        .map(PqMetalKernelVariant::from_str)
                        .collect::<Result<Vec<_>, _>>()?;
                    if cfg.variants.is_empty() {
                        return Err("--variants cannot be empty".into());
                    }
                }
                "--warmup" => cfg.warmup = parse_usize("--warmup", args.next())?,
                "--iters" => cfg.iterations = parse_usize("--iters", args.next())?,
                "--rounds" => cfg.rounds = parse_usize("--rounds", args.next())?,
                "--allow-exact-geometry" => cfg.allow_exact_geometry = true,
                "--max-dispatches" => {
                    cfg.max_dispatches = parse_usize("--max-dispatches", args.next())?
                }
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
        if !explicit_geometries.is_empty() {
            cfg.geometries = explicit_geometries;
        }
        if cfg.iterations == 0 || cfg.rounds == 0 {
            return Err("--iters and --rounds must be at least 1".into());
        }
        enforce_exact_geometry_gate(&cfg)?;
        let per_pass: usize = cfg.variants.iter().map(|v| v.dispatches_per_matvec()).sum();
        let estimated =
            cfg.geometries.len() * per_pass * (1 + cfg.rounds * (cfg.warmup + cfg.iterations));
        if estimated > cfg.max_dispatches {
            return Err(format!(
                "bounded sweep rejected: estimated {estimated} Metal dispatches exceeds \
                 --max-dispatches {}; reduce geometry/variant/round/iteration counts or \
                 raise the bound explicitly",
                cfg.max_dispatches
            ));
        }
        Ok(cfg)
    }

    fn enforce_exact_geometry_gate(cfg: &Config) -> Result<(), String> {
        let exact: Vec<_> = cfg
            .geometries
            .iter()
            .copied()
            .filter(|&(rows, cols)| census_count(rows, cols).is_some())
            .collect();
        if exact.is_empty() || cfg.allow_exact_geometry {
            return Ok(());
        }
        Err(format!(
            "manual exact-geometry gate: {exact:?} requires \
             --allow-exact-geometry; do not run the full sweep under MOP contention"
        ))
    }

    fn push_u16(out: &mut Vec<u8>, v: u16) {
        out.extend_from_slice(&v.to_le_bytes());
    }

    fn push_u32(out: &mut Vec<u8>, v: u32) {
        out.extend_from_slice(&v.to_le_bytes());
    }

    fn make_payload(rows: u32, cols: u32) -> Vec<u8> {
        let nchunk = cols / D as u32;
        let mut out = Vec::with_capacity(
            64 + CARD as usize * SUB as usize * 2 + rows as usize * nchunk as usize,
        );
        out.extend_from_slice(b"GLM52CPK");
        push_u16(&mut out, D);
        push_u16(&mut out, SUBSPACES);
        push_u16(&mut out, SUB);
        push_u16(&mut out, CARD);
        push_u32(&mut out, rows);
        push_u32(&mut out, cols);
        push_u32(&mut out, nchunk);
        push_u32(&mut out, 0xA17E_2026);
        push_u16(&mut out, BITS);
        out.push(0);
        out.push(1);
        out.resize(64, 0);
        for code in 0..CARD as usize {
            for j in 0..SUB as usize {
                let raw = ((code * 29 + j * 17 + (code ^ j) * 3) % 509) as f32;
                let v = (raw - 254.0) / 192.0;
                push_u16(&mut out, f16::from_f32(v).to_bits());
            }
        }
        for row in 0..rows as usize {
            for chunk in 0..nchunk as usize {
                let code = row
                    .wrapping_mul(73)
                    .wrapping_add(chunk.wrapping_mul(41))
                    .wrapping_add(row.wrapping_mul(chunk).wrapping_mul(3));
                out.push((code & 255) as u8);
            }
        }
        out
    }

    fn make_x(cols: usize) -> Vec<f32> {
        (0..cols)
            .map(|i| {
                let a = ((i as f32 + 0.5) * 0.017578125).sin();
                let b = ((i * 31 % 127) as f32 - 63.0) / 256.0;
                a + b
            })
            .collect()
    }

    fn timing_json(t: PqMetalTimingSummary) -> Value {
        json!({
            "min_us": t.min_us,
            "median_us": t.median_us,
            "p95_us": t.p95_us,
            "mean_us": t.mean_us,
        })
    }

    fn benchmark_json(b: &PqMetalBenchmark) -> Value {
        json!({
            "warmup": b.warmup,
            "iterations": b.iterations,
            "wall": timing_json(b.wall),
            "gpu": b.gpu.map(timing_json),
        })
    }

    fn median(mut xs: Vec<f64>) -> f64 {
        xs.sort_by(f64::total_cmp);
        xs[xs.len() / 2]
    }

    fn census_count(rows: u32, cols: u32) -> Option<u64> {
        match (rows, cols) {
            (2048, 6144) => Some(36_962),
            (6144, 2048) => Some(18_481),
            _ => None,
        }
    }

    #[derive(Debug, Clone)]
    struct CandidateGate {
        variant: String,
        parity_passed: bool,
        ranking_us: f64,
    }

    #[derive(Debug, Clone, PartialEq, Eq, Serialize)]
    struct GeometrySelection {
        parity_gate_passed: bool,
        winner: Option<String>,
    }

    /// Select only among V2.1-green candidates. An all-fail geometry is a
    /// first-class negative result, not an exceptional control-flow shortcut.
    fn select_candidate(rows: &[CandidateGate]) -> GeometrySelection {
        let winner = rows
            .iter()
            .filter(|row| row.parity_passed)
            .min_by(|a, b| a.ranking_us.total_cmp(&b.ranking_us))
            .map(|row| row.variant.clone());
        GeometrySelection {
            parity_gate_passed: winner.is_some(),
            winner,
        }
    }

    fn build_report(
        device: &str,
        estimated_dispatches: usize,
        cfg: &Config,
        geometry_reports: Vec<Value>,
    ) -> Value {
        let parity_gate_passed = geometry_reports
            .iter()
            .all(|geometry| geometry["parity_gate_passed"].as_bool() == Some(true));
        json!({
            "schema": "hawking.gravity.pq_kernel_autotune.v1",
            "status": if parity_gate_passed { "PASS" } else { "FAIL" },
            "parity_gate_passed": parity_gate_passed,
            "device": device,
            "default_unchanged": PqMetalKernelVariant::Generic.as_str(),
            "promotion": "NONE",
            "throughput_claim": "NONE; candidates remain unpromoted until a separately authorized exact-geometry run",
            "numeric_contract": "Numeric Parity V2.1; FP64 compact-matvec authority; continuous-only bounds still require exact top-k/argmax",
            "exact_geometry_gate": {
                "required_flag": "--allow-exact-geometry",
                "acknowledged": cfg.allow_exact_geometry,
                "automatic_test": false,
            },
            "bound": {
                "estimated_dispatches": estimated_dispatches,
                "max_dispatches": cfg.max_dispatches,
                "warmup": cfg.warmup,
                "iterations": cfg.iterations,
                "rounds": cfg.rounds,
            },
            "geometries": geometry_reports,
        })
    }

    /// Complete and sync the receipt before any parity-failure exit.
    fn write_receipt(path: &std::path::Path, text: &str) -> std::io::Result<()> {
        if let Some(parent) = path.parent() {
            std::fs::create_dir_all(parent)?;
        }
        let mut file = std::fs::File::create(path)?;
        file.write_all(text.as_bytes())?;
        file.write_all(b"\n")?;
        file.sync_all()
    }

    /// Emit first, then surface the parity verdict as process failure.
    fn emit_report_and_gate(
        report: &Value,
        path: &std::path::Path,
    ) -> Result<(), Box<dyn std::error::Error>> {
        let parity_gate_passed = report["parity_gate_passed"].as_bool() == Some(true);
        let text = serde_json::to_string_pretty(report)?;
        write_receipt(path, &text)?;
        println!("{}", path.display());
        if !parity_gate_passed {
            return Err(format!(
                "Numeric Parity V2.1 gate failed; negative receipt written to {}",
                path.display()
            )
            .into());
        }
        Ok(())
    }

    pub fn run() -> Result<(), Box<dyn std::error::Error>> {
        let cfg = parse_config().map_err(|e| format!("{e}\n{}", usage()))?;
        let ctx = MetalContext::new()?;
        let per_pass: usize = cfg.variants.iter().map(|v| v.dispatches_per_matvec()).sum();
        let estimated_dispatches =
            cfg.geometries.len() * per_pass * (1 + cfg.rounds * (cfg.warmup + cfg.iterations));
        eprintln!(
            "[gravity-pq-autotune] device={} geometries={} variants={} \
             warmup={} iters={} rounds={} bounded_dispatches={}/{}",
            ctx.device_name(),
            cfg.geometries.len(),
            cfg.variants.len(),
            cfg.warmup,
            cfg.iterations,
            cfg.rounds,
            estimated_dispatches,
            cfg.max_dispatches
        );

        let mut geometry_reports = Vec::new();
        for &(rows, cols) in &cfg.geometries {
            let nchunk = cols / D as u32;
            let payload = make_payload(rows, cols);
            let x = make_x(cols as usize);
            let authority = pq_matvec_f64_authority(&payload, &x)?;
            let matrix = PqMetalMatrix::from_payload(&ctx, &payload)?;
            let mut candidates = Vec::new();

            // Parity first. Pipeline compilation happens here, outside timing.
            let mut parity = Vec::new();
            for &variant in &cfg.variants {
                let output = matrix.matvec(&ctx, variant, &x)?;
                let score = score_against_f64(
                    &output,
                    &authority,
                    &Bounds::continuous_only(),
                    variant.as_str(),
                );
                eprintln!(
                    "  {rows}x{cols}/nchunk{nchunk} {:<18} parity={} rel_l2={:.3e} \
                     ulp(p50/p95/p99/max)={:.0}/{:.0}/{:.0}/{:.0}",
                    variant,
                    score.pass,
                    score.continuous.relative_l2,
                    score.continuous.ulp.median,
                    score.continuous.ulp.p95,
                    score.continuous.ulp.p99,
                    score.continuous.ulp.max,
                );
                if !score.pass {
                    if score.failures.is_empty() {
                        eprintln!("    parity failure {variant}: no detailed reason reported");
                    } else {
                        for failure in &score.failures {
                            eprintln!("    parity failure {variant}: {failure}");
                        }
                    }
                }
                parity.push((variant, score));
            }

            // Alternate candidate order each round to reduce order/thermal bias.
            let mut measurements: Vec<(PqMetalKernelVariant, Vec<PqMetalBenchmark>)> = cfg
                .variants
                .iter()
                .copied()
                .map(|v| (v, Vec::with_capacity(cfg.rounds)))
                .collect();
            for round in 0..cfg.rounds {
                let order: Vec<_> = if round % 2 == 0 {
                    cfg.variants.clone()
                } else {
                    cfg.variants.iter().rev().copied().collect()
                };
                for variant in order {
                    let b = matrix.benchmark(&ctx, variant, &x, cfg.warmup, cfg.iterations)?;
                    eprintln!(
                        "  round={} {:<18} wall_med={:.1}us gpu_med={}",
                        round + 1,
                        variant,
                        b.wall.median_us,
                        b.gpu
                            .map(|g| format!("{:.1}us", g.median_us))
                            .unwrap_or_else(|| "unavailable".into())
                    );
                    measurements
                        .iter_mut()
                        .find(|(v, _)| *v == variant)
                        .expect("registered variant")
                        .1
                        .push(b);
                }
            }

            let all_gpu = measurements
                .iter()
                .flat_map(|(_, rounds)| rounds)
                .all(|b| b.gpu.is_some());
            let timing_source = if all_gpu {
                "gpu_median_of_round_medians"
            } else {
                "wall_median_of_round_medians"
            };
            let mut gate_rows = Vec::with_capacity(measurements.len());
            for (variant, rounds) in measurements {
                let score = parity
                    .iter()
                    .find(|(v, _)| *v == variant)
                    .expect("parity result");
                let wall_median = median(rounds.iter().map(|b| b.wall.median_us).collect());
                let gpu_median = all_gpu.then(|| {
                    median(
                        rounds
                            .iter()
                            .map(|b| b.gpu.expect("all_gpu").median_us)
                            .collect(),
                    )
                });
                let ranking_us = gpu_median.unwrap_or(wall_median);
                gate_rows.push(CandidateGate {
                    variant: variant.as_str().to_string(),
                    parity_passed: score.1.pass,
                    ranking_us,
                });
                candidates.push(json!({
                    "variant": variant.as_str(),
                    "parity": score.1,
                    "rounds": rounds.iter().map(benchmark_json).collect::<Vec<_>>(),
                    "wall_median_of_round_medians_us": wall_median,
                    "gpu_median_of_round_medians_us": gpu_median,
                    "ranking_us": ranking_us,
                }));
            }
            let selection = select_candidate(&gate_rows);
            match selection.winner.as_deref() {
                Some(winner) => eprintln!(
                    "  winner {rows}x{cols}/nchunk{nchunk}: {winner} ({timing_source}); \
                     default remains generic"
                ),
                None => eprintln!(
                    "  parity gate FAILED {rows}x{cols}/nchunk{nchunk}: no V2.1-green \
                     candidate; receipt will record winner=null; default remains generic"
                ),
            }
            geometry_reports.push(json!({
                "rows": rows,
                "cols": cols,
                "nchunk": nchunk,
                "census_tensors": census_count(rows, cols),
                "payload": {
                    "kind": "deterministic synthetic bytes at sealed-artifact geometry",
                    "d": D,
                    "subspaces": SUBSPACES,
                    "sub": SUB,
                    "card": CARD,
                    "bits": BITS,
                    "bytes": payload.len(),
                },
                "timing_source": timing_source,
                "parity_gate_passed": selection.parity_gate_passed,
                "winner": selection.winner,
                "candidates": candidates,
            }));
        }

        let report = build_report(
            &ctx.device_name(),
            estimated_dispatches,
            &cfg,
            geometry_reports,
        );
        emit_report_and_gate(&report, &cfg.out)
    }

    #[cfg(test)]
    mod tests {
        use super::*;

        fn cfg() -> Config {
            Config {
                geometries: vec![(2048, 6144)],
                variants: vec![
                    PqMetalKernelVariant::Generic,
                    PqMetalKernelVariant::Bits8Vec4,
                ],
                warmup: 0,
                iterations: 1,
                rounds: 1,
                max_dispatches: 8,
                allow_exact_geometry: false,
                out: PathBuf::from("unused-by-unit-test.json"),
            }
        }

        #[test]
        fn exact_geometry_sweep_requires_explicit_manual_acknowledgement() {
            let default = Config::default();
            let per_pass: usize = default
                .variants
                .iter()
                .map(|v| v.dispatches_per_matvec())
                .sum();
            let estimated = default.geometries.len()
                * per_pass
                * (1 + default.rounds * (default.warmup + default.iterations));
            assert_eq!(estimated, default.max_dispatches);

            let mut cfg = cfg();
            let err = enforce_exact_geometry_gate(&cfg).expect_err("exact geometry must gate");
            assert!(err.contains("--allow-exact-geometry"));
            assert!(err.contains("MOP contention"));

            cfg.allow_exact_geometry = true;
            enforce_exact_geometry_gate(&cfg).expect("explicit acknowledgement");

            cfg.allow_exact_geometry = false;
            cfg.geometries = vec![(37, 2048)];
            enforce_exact_geometry_gate(&cfg).expect("tiny geometry stays bounded");
        }

        #[test]
        fn selection_picks_fastest_parity_green_candidate() {
            let rows = vec![
                CandidateGate {
                    variant: "generic".into(),
                    parity_passed: true,
                    ranking_us: 40.0,
                },
                CandidateGate {
                    variant: "bits8-direct".into(),
                    parity_passed: false,
                    ranking_us: 1.0,
                },
                CandidateGate {
                    variant: "bits8-vec4".into(),
                    parity_passed: true,
                    ranking_us: 20.0,
                },
            ];
            let selection = select_candidate(&rows);
            assert_eq!(
                selection,
                GeometrySelection {
                    parity_gate_passed: true,
                    winner: Some("bits8-vec4".into()),
                }
            );
            let report = build_report(
                "test-device",
                6,
                &cfg(),
                vec![json!({
                    "parity_gate_passed": selection.parity_gate_passed,
                    "winner": selection.winner,
                })],
            );
            assert_eq!(report["status"], "PASS");
            assert_eq!(report["parity_gate_passed"], true);
            assert_eq!(report["geometries"][0]["winner"], "bits8-vec4");
        }

        #[test]
        fn all_fail_report_keeps_scores_and_uses_null_winner() {
            let selection = select_candidate(&[
                CandidateGate {
                    variant: "generic".into(),
                    parity_passed: false,
                    ranking_us: 40.0,
                },
                CandidateGate {
                    variant: "bits8-vec4".into(),
                    parity_passed: false,
                    ranking_us: 20.0,
                },
            ]);
            assert_eq!(
                selection,
                GeometrySelection {
                    parity_gate_passed: false,
                    winner: None,
                }
            );

            let geometry = json!({
                "rows": 2048,
                "cols": 6144,
                "parity_gate_passed": selection.parity_gate_passed,
                "winner": selection.winner,
                "candidates": [
                    {
                        "variant": "generic",
                        "parity": {"pass": false, "failures": ["relative_l2"]},
                    },
                    {
                        "variant": "bits8-vec4",
                        "parity": {"pass": false, "failures": ["meaningful_rel"]},
                    },
                ],
            });
            let report = build_report("test-device", 6, &cfg(), vec![geometry]);
            assert_eq!(report["status"], "FAIL");
            assert_eq!(report["parity_gate_passed"], false);
            assert!(report["geometries"][0]["winner"].is_null());
            assert_eq!(
                report["geometries"][0]["candidates"][0]["parity"]["failures"][0],
                "relative_l2"
            );
            assert_eq!(
                report["geometries"][0]["candidates"][1]["parity"]["failures"][0],
                "meaningful_rel"
            );

            let dir = tempfile::tempdir().expect("tempdir");
            let path = dir.path().join("negative.json");
            let result = emit_report_and_gate(&report, &path);
            assert!(result.is_err(), "all-fail report must return nonzero");
            let emitted: Value =
                serde_json::from_slice(&std::fs::read(&path).expect("durable receipt"))
                    .expect("receipt JSON");
            assert_eq!(
                emitted, report,
                "emitted negative evidence must be complete"
            );
            assert!(emitted["geometries"][0]["winner"].is_null());
        }
    }
}

#[cfg(target_os = "macos")]
fn main() {
    if let Err(e) = macos::run() {
        eprintln!("gravity_pq_autotune: {e}");
        std::process::exit(2);
    }
}

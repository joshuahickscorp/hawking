//! DSV4F activation-X capture writer (reduced configuration).
//!
//! Proves the memory-bounded, deterministic per-(layer, expert) first-N
//! writer and emits the Q80/doctor6 on-disk shape. This example does **not**
//! open the sealed 148.65 GiB artifact and does **not** run a 43-layer source
//! forward. A full source capture is a separate scheduled job.
//!
//! Layers >= 2 of a real DSV4F source still need the Metal sparse-attention /
//! indexer graph, which is out of scope here. Synthetic X is used so the
//! writer, retention, flush, and collector contract can be tested.
//!
//! ```text
//! cargo run --release -p hawking-core --example dsv4f_activation_capture -- \
//!   --output-dir /abs/path/out \
//!   --tiny \
//!   --layers 3 --probes 2 --tokens-per-probe 12 \
//!   --max-hidden-tokens-per-expert 3
//! ```

use hawking_core::model::dsv4f_activation_capture::{
    default_synthetic_probes, format_capture_progress, run_reduced_capture, CaptureGeometry,
    CaptureSet, ReducedCaptureRequest, RoutePlan, DEFAULT_MAX_HIDDEN_TOKENS_PER_EXPERT,
    DEFAULT_ROW_THRESHOLD, DSV4F_CAPTURE_ROUTED_EXPERTS, DSV4F_CAPTURE_TOP_K, RESULT_SCHEMA,
};
use serde_json::json;
use std::env;
use std::path::PathBuf;
use std::process;

struct Arguments {
    output_dir: PathBuf,
    layers: usize,
    probes: usize,
    tokens_per_probe: usize,
    max_hidden_tokens_per_expert: usize,
    row_threshold: usize,
    experts: usize,
    top_k: usize,
    tiny: bool,
    capture_set: CaptureSet,
}

fn usage() -> &'static str {
    "usage: dsv4f_activation_capture --output-dir ABSOLUTE_PATH \\\n\
     \x20  [--layers N] [--probes N] [--tokens-per-probe N] \\\n\
     \x20  [--max-hidden-tokens-per-expert N] [--row-threshold N] \\\n\
     \x20  [--experts N] [--top-k N] [--tiny] [--capture-set required|doctor6]\n\
     note: --max-hidden-tokens-per-layer is refused (that budget starves 256 experts)"
}

fn fail(detail: impl AsRef<str>) -> ! {
    eprintln!("dsv4f activation capture refused: {}", detail.as_ref());
    process::exit(2);
}

fn parse_usize(value: &str, flag: &str) -> Result<usize, String> {
    value
        .parse::<usize>()
        .map_err(|_| format!("{flag} must be an unsigned decimal integer; {}", usage()))
}

fn parse_arguments() -> Result<Arguments, String> {
    let mut output_dir = None;
    let mut layers = 3usize;
    let mut probes = 2usize;
    let mut tokens_per_probe = 16usize;
    let mut max_hidden_tokens_per_expert = DEFAULT_MAX_HIDDEN_TOKENS_PER_EXPERT.min(8);
    let mut row_threshold = DEFAULT_ROW_THRESHOLD;
    let mut experts = DSV4F_CAPTURE_ROUTED_EXPERTS;
    let mut top_k = DSV4F_CAPTURE_TOP_K;
    let mut tiny = false;
    let mut capture_set = CaptureSet::required();
    let mut args = env::args().skip(1);
    while let Some(flag) = args.next() {
        match flag.as_str() {
            "--output-dir" => {
                let value = args
                    .next()
                    .ok_or_else(|| format!("missing value for --output-dir; {}", usage()))?;
                if output_dir.replace(PathBuf::from(value)).is_some() {
                    return Err("--output-dir supplied more than once".into());
                }
            }
            "--layers" => {
                let value = args
                    .next()
                    .ok_or_else(|| format!("missing value for --layers; {}", usage()))?;
                layers = parse_usize(&value, "--layers")?;
            }
            "--probes" => {
                let value = args
                    .next()
                    .ok_or_else(|| format!("missing value for --probes; {}", usage()))?;
                probes = parse_usize(&value, "--probes")?;
            }
            "--tokens-per-probe" => {
                let value = args
                    .next()
                    .ok_or_else(|| format!("missing value for --tokens-per-probe; {}", usage()))?;
                tokens_per_probe = parse_usize(&value, "--tokens-per-probe")?;
            }
            "--max-hidden-tokens-per-expert" => {
                let value = args.next().ok_or_else(|| {
                    format!(
                        "missing value for --max-hidden-tokens-per-expert; {}",
                        usage()
                    )
                })?;
                max_hidden_tokens_per_expert =
                    parse_usize(&value, "--max-hidden-tokens-per-expert")?;
            }
            "--max-hidden-tokens-per-layer" => {
                return Err(
                    "--max-hidden-tokens-per-layer is refused: a shared per-layer budget \
                     starves 256 routed experts. Use --max-hidden-tokens-per-expert N."
                        .into(),
                );
            }
            "--row-threshold" => {
                let value = args
                    .next()
                    .ok_or_else(|| format!("missing value for --row-threshold; {}", usage()))?;
                row_threshold = parse_usize(&value, "--row-threshold")?;
            }
            "--experts" => {
                let value = args
                    .next()
                    .ok_or_else(|| format!("missing value for --experts; {}", usage()))?;
                experts = parse_usize(&value, "--experts")?;
            }
            "--top-k" => {
                let value = args
                    .next()
                    .ok_or_else(|| format!("missing value for --top-k; {}", usage()))?;
                top_k = parse_usize(&value, "--top-k")?;
            }
            "--tiny" => tiny = true,
            "--capture-set" => {
                let value = args
                    .next()
                    .ok_or_else(|| format!("missing value for --capture-set; {}", usage()))?;
                capture_set = match value.as_str() {
                    "required" => CaptureSet::required(),
                    "doctor6" => CaptureSet::doctor6_only(),
                    other => {
                        return Err(format!(
                            "--capture-set must be required or doctor6 (got {other:?})"
                        ))
                    }
                };
            }
            "--help" | "-h" => {
                println!("{}", usage());
                process::exit(0);
            }
            other => return Err(format!("unknown flag {other:?}; {}", usage())),
        }
    }
    let output_dir = output_dir.ok_or_else(|| format!("missing --output-dir; {}", usage()))?;
    if !output_dir.is_absolute() {
        return Err("--output-dir must be an absolute path".into());
    }
    if tiny {
        if experts == DSV4F_CAPTURE_ROUTED_EXPERTS {
            experts = 16;
        }
    }
    Ok(Arguments {
        output_dir,
        layers,
        probes,
        tokens_per_probe,
        max_hidden_tokens_per_expert,
        row_threshold,
        experts,
        top_k,
        tiny,
        capture_set,
    })
}

fn main() {
    let arguments = parse_arguments().unwrap_or_else(|e| fail(e));
    let mut geometry = if arguments.tiny {
        CaptureGeometry::test_fixture()
    } else {
        CaptureGeometry::sealed()
    };
    geometry.layers = arguments.layers;
    geometry.routed_experts = arguments.experts;
    geometry.top_k = arguments.top_k;
    if arguments.tiny {
        // test_fixture already set tiny widths; keep them.
    }
    let probes = default_synthetic_probes(arguments.probes, arguments.tokens_per_probe);
    let total_tokens: usize = probes.iter().map(|(_, t)| t.len()).sum();
    eprintln!(
        "{}",
        format_capture_progress(
            probes.len(),
            total_tokens,
            &geometry,
            arguments.max_hidden_tokens_per_expert,
        )
    );
    let report = run_reduced_capture(&ReducedCaptureRequest {
        output_dir: arguments.output_dir.clone(),
        geometry,
        set: arguments.capture_set,
        probes,
        max_per_expert: arguments.max_hidden_tokens_per_expert,
        row_threshold: arguments.row_threshold,
        route_plan: RoutePlan::Cycling,
    })
    .unwrap_or_else(|e| fail(e.to_string()));
    let summary = json!({
        "status": "ok",
        "schema": RESULT_SCHEMA,
        "output_dir": arguments.output_dir,
        "result_path": report.result_path,
        "hidden_rows_retained_total": report.hidden_rows_retained_total,
        "hidden_bytes_written": report.hidden_bytes_written,
        "peak_resident_bytes": report.peak_resident_bytes,
        "resident_after_append": report.resident_after_append,
        "n_fit_distribution": report.n_fit,
        "honesty": {
            "synthetic_activations": true,
            "sealed_artifact_not_opened": true,
            "full_43_layer_source_forward_not_executed": true,
            "layers_ge_2_need_metal_sparse_attention_for_a_real_source": true,
        },
    });
    println!(
        "{}",
        serde_json::to_string_pretty(&summary).expect("summary serializes")
    );
}

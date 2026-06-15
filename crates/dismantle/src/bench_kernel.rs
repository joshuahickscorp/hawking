use anyhow::Result;
use dismantle_core::kernel_bench::{self, ALL_KERNEL_NAMES};
use std::io::Write;

pub struct BenchKernelOptions {
    pub kernel: Option<String>,
    pub all: bool,
    pub shape: String,
    pub iterations: usize,
    pub no_history: bool,
}

pub fn run(opts: BenchKernelOptions) -> Result<()> {
    let (rows, cols) = kernel_bench::parse_shape(&opts.shape)?;

    let kernels: Vec<&str> = if opts.all {
        ALL_KERNEL_NAMES.to_vec()
    } else if let Some(ref k) = opts.kernel {
        vec![k.as_str()]
    } else {
        anyhow::bail!("provide --kernel <name> or --all");
    };

    let commit = git_short_head();
    let mut results = Vec::new();

    for kernel_name in &kernels {
        eprint!(
            "  benching {} at {}x{} ({} iters)... ",
            kernel_name, rows, cols, opts.iterations
        );
        let _ = std::io::stderr().flush();

        match kernel_bench::run_kernel(kernel_name, rows, cols, opts.iterations) {
            Ok(result) => {
                eprintln!(
                    "mean={:.1}μs p50={:.1}μs p99={:.1}μs",
                    result.mean_us, result.p50_us, result.p99_us
                );
                println!("{}", serde_json::to_string(&result)?);
                results.push(result);
            }
            Err(e) => {
                eprintln!("SKIP: {e}");
            }
        }
    }

    if results.is_empty() {
        anyhow::bail!("no kernels ran successfully at shape {}x{}", rows, cols);
    }

    if !opts.no_history {
        append_to_history(&results, &commit)?;
    }

    Ok(())
}

fn git_short_head() -> String {
    std::process::Command::new("git")
        .args(["rev-parse", "--short", "HEAD"])
        .output()
        .ok()
        .and_then(|o| String::from_utf8(o.stdout).ok())
        .map(|s| s.trim().to_string())
        .filter(|s| !s.is_empty())
        .unwrap_or_else(|| "unknown".into())
}

fn append_to_history(
    results: &[dismantle_core::kernel_bench::KernelBenchResult],
    _commit: &str,
) -> Result<()> {
    std::fs::create_dir_all("bench_results")?;
    let mut f = std::fs::OpenOptions::new()
        .append(true)
        .create(true)
        .open("bench_results/kernel_perf_history.jsonl")?;
    for r in results {
        writeln!(f, "{}", serde_json::to_string(r)?)?;
    }
    eprintln!(
        "appended {} result(s) to bench_results/kernel_perf_history.jsonl",
        results.len()
    );
    Ok(())
}

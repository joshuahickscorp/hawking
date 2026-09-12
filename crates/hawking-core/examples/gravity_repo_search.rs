//! Run the deterministic native Gravity repository search.
//!
//! Usage:
//! `cargo run -p hawking-core --example gravity_repo_search -- /path/to/repo needle [repeats]`

use hawking_core::gravity_repo_search::{RepoSearchIndex, RepoSearchLimits};
use serde_json::json;
use std::{env, process::ExitCode, time::Instant};

fn main() -> ExitCode {
    let mut arguments = env::args_os().skip(1);
    let (Some(root), Some(needle)) = (arguments.next(), arguments.next()) else {
        eprintln!("usage: gravity_repo_search <root> <needle> [repeats]");
        return ExitCode::from(2);
    };
    let repeats = arguments
        .next()
        .and_then(|value| value.into_string().ok())
        .and_then(|value| value.parse::<usize>().ok())
        .filter(|value| (1..=100).contains(value))
        .unwrap_or(5);
    let root_display = root.to_string_lossy().into_owned();
    let needle_display = needle.to_string_lossy().into_owned();
    let limits = RepoSearchLimits::default();
    let build_started = Instant::now();
    let index = match RepoSearchIndex::build(&root, limits) {
        Ok(index) => index,
        Err(error) => {
            eprintln!("native repository index refused: {error}");
            return ExitCode::from(1);
        }
    };
    let index_build_ns = build_started.elapsed().as_nanos();
    let mut samples = Vec::with_capacity(repeats);
    let mut latest = None;
    for _ in 0..repeats {
        let started = Instant::now();
        match index.search(&needle_display, limits.max_results) {
            Ok(result) => {
                samples.push(started.elapsed().as_nanos());
                latest = Some(result);
            }
            Err(error) => {
                eprintln!("native repository search refused: {error}");
                return ExitCode::from(1);
            }
        }
    }
    samples.sort();
    let result = latest.expect("at least one successful repeat");
    println!(
        "{}",
        serde_json::to_string_pretty(&json!({
            "schema": "hawking.gravity.native_repo_search.v1",
            "root": root_display,
            "needle": needle_display,
            "repeats": repeats,
            "timing": {
                "index_build_ns": index_build_ns,
                "min_ns": samples[0],
                "median_ns": samples[(samples.len() - 1) / 2],
                "p95_ns": samples[(samples.len() - 1) * 95 / 100],
                "max_ns": samples[samples.len() - 1],
            },
            "result": {
                "files_seen": result.files_seen,
                "indexed_files": index.indexed_files(),
                "skipped_large": result.skipped_large,
                "matches": result.matches.len(),
                "truncated": result.truncated,
                "first_matches": result.matches.iter().take(8).map(|row| json!({
                    "path": row.path,
                    "line": row.line,
                    "text": row.text,
                })).collect::<Vec<_>>(),
            }
        }))
        .expect("serializing primitive search output cannot fail")
    );
    ExitCode::SUCCESS
}

//! CLI surface for `hawking-index`.
//!
//! Subcommands emit JSON on stdout so Python (and anything else) can consume
//! index facts without a pyo3 extension or a shared ABI.

use clap::{Parser, Subcommand};
use hawking_index::reachability::{collect_reachability_facts, CollectOptions};
use std::path::PathBuf;
use std::process::ExitCode;

#[derive(Parser, Debug)]
#[command(name = "hawking-index", about = "Living codebase index")]
struct Cli {
    #[command(subcommand)]
    cmd: Cmd,
}

#[derive(Subcommand, Debug)]
enum Cmd {
    /// Emit Python reachability facts (imports, binds, calls, subprocess,
    /// tool-dispatch literals) as JSON on stdout.
    ///
    /// Walks `*.py` blobs at `--commit` (default HEAD), never the working
    /// tree. Merkle-diffs against a local cache and reparses only blobs
    /// whose SHA changed, using the same tree-sitter Python grammar the
    /// rest of hawking-index uses.
    ReachabilityFacts {
        /// Repository root (default: cwd).
        #[arg(long)]
        root: Option<PathBuf>,
        /// Git revision to read (default HEAD). Ignored if `root` is not a repo.
        #[arg(long, default_value = "HEAD")]
        commit: String,
        /// Cache directory (default: `<root>/.hide/reachability-index`).
        #[arg(long)]
        cache: Option<PathBuf>,
        /// Pretty-print JSON.
        #[arg(long)]
        pretty: bool,
        /// Write JSON to this path instead of stdout (Python reads the file
        /// so a  tens-of-MB dump does not traverse a pipe).
        #[arg(long)]
        output: Option<PathBuf>,
    },
}

fn main() -> ExitCode {
    let cli = Cli::parse();
    match cli.cmd {
        Cmd::ReachabilityFacts {
            root,
            commit,
            cache,
            pretty,
            output,
        } => {
            let root = root
                .unwrap_or_else(|| std::env::current_dir().unwrap_or_else(|_| PathBuf::from(".")));
            let mut opts = CollectOptions::new(&root).with_commit(commit);
            if let Some(cache) = cache {
                opts = opts.with_cache_dir(cache);
            } else if let Ok(env) = std::env::var("HAWKING_INDEX_CACHE") {
                if !env.is_empty() {
                    opts = opts.with_cache_dir(PathBuf::from(env));
                }
            }
            match collect_reachability_facts(&opts) {
                Ok(dump) => {
                    // Warm hit: facts.json is already the dump. Skip a 40MB
                    // serde+write that dominated assemble() wall time.
                    if let Some(path) = output.as_ref() {
                        if dump.index.parsed == 0 && !dump.index.cold && path.is_file() {
                            return ExitCode::SUCCESS;
                        }
                    }
                    let json = if pretty {
                        serde_json::to_string_pretty(&dump)
                    } else {
                        serde_json::to_string(&dump)
                    };
                    match json {
                        Ok(s) => {
                            if let Some(path) = output {
                                if let Some(parent) = path.parent() {
                                    if let Err(e) = std::fs::create_dir_all(parent) {
                                        eprintln!("hawking-index: {e}");
                                        return ExitCode::from(1);
                                    }
                                }
                                if let Err(e) = std::fs::write(&path, s) {
                                    eprintln!("hawking-index: write {}: {e}", path.display());
                                    return ExitCode::from(1);
                                }
                            } else {
                                println!("{s}");
                            }
                            ExitCode::SUCCESS
                        }
                        Err(e) => {
                            eprintln!("hawking-index: serialize: {e}");
                            ExitCode::from(2)
                        }
                    }
                }
                Err(e) => {
                    eprintln!("hawking-index: {e}");
                    ExitCode::from(1)
                }
            }
        }
    }
}

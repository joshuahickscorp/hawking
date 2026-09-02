//! STATIC_ONLY host/shader ABI preflight CLI.
//!
//! ```text
//! cargo run --release -p hawking-core --bin hawking-static-kernel-verify -- --repo . --json
//! ```
//!
//! Prints the same JSON document as `tools/future/static_kernel_verify.scan()`.
//! Not a speed claim. Absent this binary, the Python path remains the default.

use hawking_core::static_kernel_verify::scan_repo;
use std::env;
use std::io::{self, Write};
use std::path::PathBuf;
use std::process;

fn usage() -> &'static str {
    "usage: hawking-static-kernel-verify [--repo PATH] [--json]\n\
     STATIC_ONLY host/shader ABI preflight. JSON on stdout. Never a hardware measurement."
}

fn main() {
    let mut repo = env::current_dir().unwrap_or_else(|_| PathBuf::from("."));
    let mut json_out = true;
    let mut args = env::args().skip(1);
    while let Some(a) = args.next() {
        match a.as_str() {
            "--repo" => {
                let Some(p) = args.next() else {
                    eprintln!("{}", usage());
                    process::exit(2);
                };
                repo = PathBuf::from(p);
            }
            "--json" => json_out = true,
            "-h" | "--help" => {
                println!("{}", usage());
                return;
            }
            other => {
                eprintln!("unknown argument: {other}\n{}", usage());
                process::exit(2);
            }
        }
    }
    let repo = match repo.canonicalize() {
        Ok(p) => p,
        Err(e) => {
            eprintln!("cannot canonicalize repo {}: {e}", repo.display());
            process::exit(1);
        }
    };
    let doc = scan_repo(&repo);
    if json_out {
        match serde_json::to_string(&doc) {
            Ok(s) => {
                let mut out = io::stdout().lock();
                let _ = writeln!(out, "{s}");
            }
            Err(e) => {
                eprintln!("json encode failed: {e}");
                process::exit(1);
            }
        }
    }
}

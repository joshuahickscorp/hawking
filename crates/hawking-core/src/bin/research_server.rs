//! Persistent local research server for Q80 / DSV4F dirty Tier-1 loops.
//!
//! ```text
//! cargo run --release -p hawking-core --bin research_server
//! cargo run --release -p hawking-core --bin research_server -- --socket /tmp/hawking-research.sock
//! ```
//!
//! Protocol (one command per line, one JSON reply per line):
//!   load q80 [path]
//!   load dsv4f [path] [--metal]
//!   greedy <prompt> <n_tokens>
//!   time <n_tokens>
//!   reload-kernels
//!   status
//!   quit
//!
//! Every reply is labelled DIRTY_TIER1. Not BASE_TRUE_TPS. Local only:
//! stdin/stdout or a unix-domain socket. No TCP listener.

use hawking_core::research_server::{serve_stdio, serve_unix_socket, ResearchServer, TIMING_CLASS};
use std::env;
use std::io::{self, BufReader};
use std::path::PathBuf;
use std::process;

fn usage() -> &'static str {
    "usage: research_server [--socket PATH] [--stdin]"
}

fn main() {
    if let Err(error) = run() {
        eprintln!("{error}");
        process::exit(1);
    }
}

fn run() -> Result<(), String> {
    hawking_core::startup_timing::mark_process_start();
    let mut socket = None;
    let mut args = env::args().skip(1);
    while let Some(flag) = args.next() {
        match flag.as_str() {
            "--socket" => {
                socket = Some(PathBuf::from(
                    args.next().ok_or_else(|| usage().to_owned())?,
                ));
            }
            "--stdin" => socket = None,
            "--help" | "-h" => return Err(usage().to_owned()),
            other => return Err(format!("unknown flag {other}; {}", usage())),
        }
    }

    let mut server = ResearchServer::new();
    eprintln!(
        "research-server ready timing_class={TIMING_CLASS} transport={}",
        if socket.is_some() {
            "unix-socket"
        } else {
            "stdio"
        }
    );
    if let Some(path) = socket {
        serve_unix_socket(&mut server, &path).map_err(|error| error.to_string())?;
    } else {
        let stdin = io::stdin();
        let stdout = io::stdout();
        let _quit = serve_stdio(&mut server, BufReader::new(stdin.lock()), stdout.lock())
            .map_err(|error| error.to_string())?;
    }
    Ok(())
}

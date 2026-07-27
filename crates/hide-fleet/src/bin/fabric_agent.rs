//! Fabric Agent process — registers, heartbeats, accepts placement, reports failure.
//!
//! ```text
//! fabric-agent serve --node-id NODE --listen 127.0.0.1:PORT
//! ```
//!
//! Localhost only. ABI does not assume co-location with the coordinator.

use hide_fleet::fabric::agent::{AgentConfig, FabricAgent};
use hide_fleet::fabric::protocol::{AgentRequest, AgentResponse};
use std::env;
use std::io::{BufRead, BufReader, Write};
use std::net::TcpListener;
use std::sync::Arc;

fn main() {
    let mut args = env::args().skip(1).collect::<Vec<_>>();
    if args.is_empty() {
        eprintln!("usage: fabric-agent serve --node-id ID --listen ADDR");
        std::process::exit(2);
    }
    let cmd = args.remove(0);
    if cmd != "serve" {
        eprintln!("unknown command {cmd}");
        std::process::exit(2);
    }
    let mut node_id = "local".to_string();
    let mut listen = "127.0.0.1:0".to_string();
    let mut i = 0;
    while i < args.len() {
        match args[i].as_str() {
            "--node-id" => {
                i += 1;
                node_id = args.get(i).cloned().unwrap_or_else(|| {
                    eprintln!("missing --node-id value");
                    std::process::exit(2);
                });
            }
            "--listen" => {
                i += 1;
                listen = args.get(i).cloned().unwrap_or_else(|| {
                    eprintln!("missing --listen value");
                    std::process::exit(2);
                });
            }
            other => {
                eprintln!("unknown arg {other}");
                std::process::exit(2);
            }
        }
        i += 1;
    }

    let agent = Arc::new(FabricAgent::new(AgentConfig::new(node_id, listen.clone())));
    let listener = TcpListener::bind(&listen).unwrap_or_else(|e| {
        eprintln!("bind {listen}: {e}");
        std::process::exit(1);
    });
    // Single-threaded accept loop is enough for the software fixture.
    for conn in listener.incoming() {
        if !agent.is_running() {
            break;
        }
        let Ok(mut stream) = conn else {
            continue;
        };
        let mut reader = BufReader::new(stream.try_clone().expect("clone"));
        let mut line = String::new();
        if reader.read_line(&mut line).ok().unwrap_or(0) == 0 {
            continue;
        }
        let resp = match serde_json::from_str::<AgentRequest>(line.trim()) {
            Ok(req) => agent.handle(req),
            Err(e) => AgentResponse::Error {
                message: format!("bad request: {e}"),
            },
        };
        if let Ok(body) = serde_json::to_string(&resp) {
            let _ = stream.write_all(body.as_bytes());
            let _ = stream.write_all(b"\n");
            let _ = stream.flush();
        }
        if matches!(resp, AgentResponse::Ok { detail, .. } if detail == "shutdown") {
            break;
        }
    }
}

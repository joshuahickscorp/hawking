//! CLI over [`hawking_index::artifact`]: index / get / list / fresh / parity.
//!
//! Does not rewrite the source JSON. The sidecar is derived.

use hawking_index::artifact::{parity_map, ArtifactIndex};
use std::env;
use std::fs;
use std::path::PathBuf;
use std::process::ExitCode;

fn usage() -> ! {
    eprintln!(
        "hawking-artifact — queryable sidecar over a JSON object-map receipt

  index  --input JSON --output SQLITE [--maps modules,gates,genes]
  get    --index SQLITE --map MAP --key KEY
  list   --index SQLITE --map MAP [--classification C] [--disposition D] [--status S]
  fresh  --index SQLITE --input JSON
  parity --index SQLITE --input JSON --map MAP
  meta   --index SQLITE

The source JSON is the durable record. This tool never deletes or rewrites it."
    );
    std::process::exit(2);
}

fn flag(args: &[String], name: &str) -> Option<String> {
    let mut i = 0;
    while i < args.len() {
        if args[i] == name {
            return args.get(i + 1).cloned();
        }
        if let Some(rest) = args[i].strip_prefix(&format!("{name}=")) {
            return Some(rest.to_string());
        }
        i += 1;
    }
    None
}

fn req(args: &[String], name: &str) -> String {
    flag(args, name).unwrap_or_else(|| {
        eprintln!("missing {name}");
        usage();
    })
}

fn main() -> ExitCode {
    let args: Vec<String> = env::args().skip(1).collect();
    if args.is_empty() || args[0] == "-h" || args[0] == "--help" {
        usage();
    }
    let cmd = args[0].as_str();
    let rest = &args[1..];
    match cmd {
        "index" => cmd_index(rest),
        "get" => cmd_get(rest),
        "list" => cmd_list(rest),
        "fresh" => cmd_fresh(rest),
        "parity" => cmd_parity(rest),
        "meta" => cmd_meta(rest),
        _ => {
            eprintln!("unknown command {cmd}");
            usage();
        }
    }
}

fn cmd_index(args: &[String]) -> ExitCode {
    let input = PathBuf::from(req(args, "--input"));
    let output = PathBuf::from(req(args, "--output"));
    let maps_owned: Vec<String> = flag(args, "--maps")
        .unwrap_or_default()
        .split(',')
        .map(|s| s.trim().to_string())
        .filter(|s| !s.is_empty())
        .collect();
    let maps: Vec<&str> = maps_owned.iter().map(|s| s.as_str()).collect();
    match ArtifactIndex::build(&input, &output, &maps) {
        Ok(idx) => match idx.meta() {
            Ok(m) => {
                println!(
                    "{}",
                    serde_json::json!({
                        "ok": true,
                        "schema": m.schema,
                        "source_sha256": m.source_sha256,
                        "source_size": m.source_size,
                        "n_entities": m.n_entities,
                        "index": output.display().to_string(),
                    })
                );
                ExitCode::SUCCESS
            }
            Err(e) => fail(&e),
        },
        Err(e) => fail(&e),
    }
}

fn cmd_get(args: &[String]) -> ExitCode {
    let index = PathBuf::from(req(args, "--index"));
    let map = req(args, "--map");
    let key = req(args, "--key");
    match ArtifactIndex::open(&index).and_then(|idx| idx.get_json(&map, &key)) {
        Ok(Some(raw)) => {
            print!("{raw}");
            ExitCode::SUCCESS
        }
        Ok(None) => {
            eprintln!("not found: {map} {key}");
            ExitCode::from(1)
        }
        Err(e) => fail(&e),
    }
}

fn cmd_list(args: &[String]) -> ExitCode {
    let index = PathBuf::from(req(args, "--index"));
    let map = req(args, "--map");
    let idx = match ArtifactIndex::open(&index) {
        Ok(i) => i,
        Err(e) => return fail(&e),
    };
    let result = if let Some(c) = flag(args, "--classification") {
        idx.keys_where(&map, "classification", &c)
    } else if let Some(d) = flag(args, "--disposition") {
        idx.keys_where(&map, "disposition", &d)
    } else if let Some(s) = flag(args, "--status") {
        idx.keys_where(&map, "status", &s)
    } else {
        idx.keys(&map)
    };
    match result {
        Ok(keys) => {
            println!("{}", serde_json::to_string(&keys).unwrap());
            ExitCode::SUCCESS
        }
        Err(e) => fail(&e),
    }
}

fn cmd_fresh(args: &[String]) -> ExitCode {
    let index = PathBuf::from(req(args, "--index"));
    let input = PathBuf::from(req(args, "--input"));
    match ArtifactIndex::open(&index).and_then(|idx| idx.is_fresh(&input)) {
        Ok(true) => {
            println!("{{\"fresh\":true}}");
            ExitCode::SUCCESS
        }
        Ok(false) => {
            println!("{{\"fresh\":false}}");
            ExitCode::from(1)
        }
        Err(e) => fail(&e),
    }
}

fn cmd_parity(args: &[String]) -> ExitCode {
    let index = PathBuf::from(req(args, "--index"));
    let input = PathBuf::from(req(args, "--input"));
    let map = req(args, "--map");
    let bytes = match fs::read(&input) {
        Ok(b) => b,
        Err(e) => {
            eprintln!("{e}");
            return ExitCode::from(1);
        }
    };
    match ArtifactIndex::open(&index).and_then(|idx| parity_map(&idx, &bytes, &map)) {
        Ok((n_equal, mismatches)) => {
            let ok = mismatches.is_empty();
            println!(
                "{}",
                serde_json::json!({
                    "ok": ok,
                    "n_equal": n_equal,
                    "n_mismatch": mismatches.len(),
                    "mismatches": mismatches.iter().take(20).cloned().collect::<Vec<_>>(),
                })
            );
            if ok {
                ExitCode::SUCCESS
            } else {
                ExitCode::from(1)
            }
        }
        Err(e) => fail(&e),
    }
}

fn cmd_meta(args: &[String]) -> ExitCode {
    let index = PathBuf::from(req(args, "--index"));
    match ArtifactIndex::open(&index).and_then(|idx| idx.meta()) {
        Ok(m) => {
            println!("{}", serde_json::to_value(&m).unwrap());
            ExitCode::SUCCESS
        }
        Err(e) => fail(&e),
    }
}

fn fail(e: &impl std::fmt::Display) -> ExitCode {
    eprintln!("{e}");
    ExitCode::from(1)
}

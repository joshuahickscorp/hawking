//! Native Qwen3.8 greedy decode of an HGRAVF01 affine2 catalog.
//!
//! FOREIGN-RUNTIME: Hawking Metal only. Does not spawn or call mlx / mlx_lm.
//!
//! ```text
//! cargo build -p hawking-core --example affine2_native_decode
//! ./target/release/examples/affine2_native_decode \
//!   --artifact-root workspace/campaign/records/runs/qwen38-27b/affine2-native \
//!   --tokenizer .../abliterated-mlx-2bit/2bit/tokenizer.json \
//!   --prompt "The capital of France is" --raw-prompt --max-new-tokens 16
//! ```

use hawking_core::model::qwen38_hybrid_decode::{
    load_qwen38_tokenizer, render_qwen38_user_chat, Qwen38GenerateResult,
};
use std::env;
use std::path::{Path, PathBuf};
use std::process;

#[cfg(target_os = "macos")]
use hawking_core::model::qwen38_hybrid_decode::{generate_greedy, Qwen38HybridDecodeSession};

fn usage() -> &'static str {
    "usage: affine2_native_decode [--artifact-root DIR] [--tokenizer PATH] \
        [--prompt TEXT] [--raw-prompt] [--max-new-tokens N] [--max-seq-len N]"
}

fn fail(message: impl std::fmt::Display) -> ! {
    eprintln!("affine2_native_decode: {message}");
    process::exit(2);
}

fn repo_root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../..")
}

fn default_artifact() -> PathBuf {
    let local = repo_root().join("workspace/campaign/records/runs/qwen38-27b/affine2-native");
    if local.join("catalog.hq38m20").is_file() {
        return local;
    }
    PathBuf::from(
        "/Users/scammermike/Downloads/hawking/workspace/campaign/records/runs/qwen38-27b/affine2-native",
    )
}

fn default_tokenizer(artifact: &Path) -> PathBuf {
    let beside = artifact.join("tokenizer.json");
    if beside.is_file() {
        return beside;
    }
    PathBuf::from(
        "/Users/scammermike/Downloads/hawking/workspace/campaign/records/runs/qwen38-27b/abliterated-mlx-2bit/2bit/tokenizer.json",
    )
}

struct Args {
    artifact_root: PathBuf,
    tokenizer: PathBuf,
    prompt: String,
    raw_prompt: bool,
    max_new_tokens: usize,
    max_seq_len: usize,
}

fn parse_args() -> Args {
    let mut artifact_root = None;
    let mut tokenizer = None;
    let mut prompt = "The capital of France is".to_owned();
    let mut raw_prompt = true;
    let mut max_new_tokens = 16usize;
    let mut max_seq_len = 128usize;
    let mut args = env::args().skip(1);
    while let Some(flag) = args.next() {
        match flag.as_str() {
            "--artifact-root" => {
                artifact_root = Some(PathBuf::from(args.next().unwrap_or_else(|| fail(usage()))));
            }
            "--tokenizer" => {
                tokenizer = Some(PathBuf::from(args.next().unwrap_or_else(|| fail(usage()))));
            }
            "--prompt" => prompt = args.next().unwrap_or_else(|| fail(usage())),
            "--raw-prompt" => raw_prompt = true,
            "--chat-prompt" => raw_prompt = false,
            "--max-new-tokens" => {
                max_new_tokens = args
                    .next()
                    .unwrap_or_else(|| fail(usage()))
                    .parse()
                    .unwrap_or_else(|_| fail("--max-new-tokens"));
            }
            "--max-seq-len" => {
                max_seq_len = args
                    .next()
                    .unwrap_or_else(|| fail(usage()))
                    .parse()
                    .unwrap_or_else(|_| fail("--max-seq-len"));
            }
            other => fail(format!("unknown {other}; {}", usage())),
        }
    }
    let artifact_root = artifact_root.unwrap_or_else(default_artifact);
    let tokenizer = tokenizer.unwrap_or_else(|| default_tokenizer(&artifact_root));
    Args {
        artifact_root,
        tokenizer,
        prompt,
        raw_prompt,
        max_new_tokens,
        max_seq_len,
    }
}

fn decode_tps(result: &Qwen38GenerateResult) -> f64 {
    match result.steady_decode_wall_ns_per_token() {
        Some(ns) if ns > 0 => 1.0e9 / ns as f64,
        _ => 0.0,
    }
}

#[cfg(not(target_os = "macos"))]
fn main() {
    fail("affine2 native decode is Metal-only");
}

#[cfg(target_os = "macos")]
fn main() {
    let args = parse_args();
    if !args.artifact_root.join("catalog.hq38m20").is_file() {
        fail(format!(
            "missing {} — run tools/affine2_transcode.py first",
            args.artifact_root.join("catalog.hq38m20").display()
        ));
    }
    let tokenizer = load_qwen38_tokenizer(&args.tokenizer).unwrap_or_else(|e| fail(e));
    let rendered = if args.raw_prompt {
        args.prompt.clone()
    } else {
        render_qwen38_user_chat(&args.prompt)
    };
    let prompt_ids = tokenizer
        .encode(&rendered, false)
        .unwrap_or_else(|e| fail(e));
    eprintln!(
        "affine2_native_decode artifact={} prompt_len={} raw={} FOREIGN-RUNTIME=hawking-metal (no mlx)",
        args.artifact_root.display(),
        prompt_ids.len(),
        args.raw_prompt
    );
    let mut session = Qwen38HybridDecodeSession::open(&args.artifact_root, args.max_seq_len)
        .unwrap_or_else(|e| fail(e));
    let result = generate_greedy(&mut session, &prompt_ids, args.max_new_tokens)
        .unwrap_or_else(|e| fail(e));
    let text = result.decode_new(&tokenizer).unwrap_or_else(|e| fail(e));
    let tps = decode_tps(&result);
    println!("FOREIGN-RUNTIME: hawking-metal");
    println!("FOREIGN-RUNTIME-MLX: 0");
    println!("GENERATED_TEXT: {text}");
    println!("GENERATED_TEXT_VERBATIM: {text}");
    println!("NEW_TOKENS: {:?}", result.new_tokens());
    println!("PROMPT_LEN: {}", result.prompt_len);
    println!("DECODE_STEPS: {}", result.decode_steps);
    println!("DECODE_WALL_NS: {}", result.decode_wall_ns);
    println!(
        "STEADY_DECODE_WALL_NS_PER_TOKEN: {:?}",
        result.steady_decode_wall_ns_per_token()
    );
    println!("DECODE_TPS: {tps:.4}");
    println!("FALLBACKS: {}", result.fallbacks);
    println!("DENSE_W_MATERIALIZED: 0");
    if result.fallbacks != 0 {
        fail(format!("fallbacks={}", result.fallbacks));
    }
}

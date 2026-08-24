//! Runtime dispatch-name probe for Qwen3.8 hybrid decode.
//!
//! Opens a catalog, generates a few tokens, and prints the distinct
//! kernel names the runtime actually dispatched. Names come from the
//! TCB structural label list (exact `dispatch_threads` strings), not
//! from NX source-literal extraction.
//!
//! ```text
//! HAWKING_TRACE_DISPATCH=1 \
//!   ./workspace/ops/build/rust/release-fast/examples/qwen38_dispatch_trace \
//!   --artifact-root .../uniform-q4-v1 \
//!   --tokenizer .../bf16/tokenizer.json
//! ```

use std::env;
use std::path::PathBuf;
use std::process;

#[cfg(target_os = "macos")]
use hawking_core::model::qwen38_hybrid_decode::{
    generate_greedy, load_qwen38_tokenizer, render_qwen38_user_chat, Qwen38HybridDecodeSession,
};

fn usage() -> &'static str {
    "usage: qwen38_dispatch_trace --artifact-root DIR --tokenizer PATH \
        [--prompt TEXT] [--raw-prompt] [--max-new-tokens N] [--max-seq-len N]"
}

fn fail(message: impl std::fmt::Display) -> ! {
    eprintln!("qwen38_dispatch_trace: {message}");
    process::exit(2);
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
    let mut prompt = "Say hi.".to_owned();
    let mut raw_prompt = false;
    let mut max_new_tokens = 4usize;
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
    Args {
        artifact_root: artifact_root.unwrap_or_else(|| fail(usage())),
        tokenizer: tokenizer.unwrap_or_else(|| fail(usage())),
        prompt,
        raw_prompt,
        max_new_tokens,
        max_seq_len,
    }
}

#[cfg(not(target_os = "macos"))]
fn main() {
    fail("qwen38 dispatch trace is Metal-only");
}

#[cfg(target_os = "macos")]
fn main() {
    // Structural TCB names require Off. Cpu/gpu timing remaps through
    // static_kernel_name (several bound kernels currently become "other").
    env::set_var("HAWKING_TRACE_DISPATCH", "1");
    env::set_var("HAWKING_TCB_TRACE", "0");

    let args = parse_args();
    if args.max_new_tokens == 0 {
        fail("--max-new-tokens must be >= 1");
    }
    if args.max_seq_len == 0 {
        fail("--max-seq-len must be >= 1");
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
    if prompt_ids.is_empty() {
        fail("prompt encoded to zero tokens");
    }
    if prompt_ids.len() + args.max_new_tokens > args.max_seq_len {
        fail(format!(
            "prompt_len={} + max_new_tokens={} exceeds max_seq_len={}",
            prompt_ids.len(),
            args.max_new_tokens,
            args.max_seq_len
        ));
    }

    eprintln!(
        "qwen38-dispatch-trace artifact={} prompt_len={} max_new={}",
        args.artifact_root.display(),
        prompt_ids.len(),
        args.max_new_tokens
    );
    let mut session = Qwen38HybridDecodeSession::open(&args.artifact_root, args.max_seq_len)
        .unwrap_or_else(|e| fail(e));
    let result =
        generate_greedy(&mut session, &prompt_ids, args.max_new_tokens).unwrap_or_else(|e| fail(e));
    if result.fallbacks != 0 {
        fail(format!("fallbacks={}", result.fallbacks));
    }
    let names = session.drain_dispatched_kernel_names();
    println!("ARTIFACT: {}", args.artifact_root.display());
    println!("PROMPT_LEN: {}", result.prompt_len);
    println!("NEW_TOKENS: {:?}", result.new_tokens());
    println!("FALLBACKS: {}", result.fallbacks);
    println!("DISPATCHED_COUNT: {}", names.len());
    println!("DISPATCHED_KERNELS:");
    for name in &names {
        println!("{name}");
    }
}

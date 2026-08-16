//! Native Qwen3.8 greedy generate on the language-only Q4 catalog.
//!
//! ```text
//! ./tools/gpu_lane_lock.sh qwen38-native-bringup \
//!   workspace/ops/build/rust/release-fast/examples/ascension_qwen38_hybrid_greedy \
//!   --artifact-root .../uniform-q4-v1 \
//!   --tokenizer .../bf16/tokenizer.json \
//!   --prompt "Say hi." --max-new-tokens 16 \
//!   --out receipts/ascent-2026-08-16/qwen38-native-generate.json
//! ```

use hawking_core::model::qwen38_hybrid_decode::{
    load_qwen38_tokenizer, render_qwen38_user_chat, Qwen38GenerateResult,
};
use serde_json::json;
use std::env;
use std::fs;
use std::path::PathBuf;
use std::process;

#[cfg(target_os = "macos")]
use hawking_core::model::qwen38_hybrid_decode::{generate_greedy, Qwen38HybridDecodeSession};

fn usage() -> &'static str {
    "usage: ascension_qwen38_hybrid_greedy --artifact-root DIR --tokenizer PATH \
        [--prompt TEXT] [--raw-prompt] [--max-new-tokens N] [--max-seq-len N] [--out FILE]"
}

fn fail(message: impl std::fmt::Display) -> ! {
    eprintln!("ascension_qwen38_hybrid_greedy: {message}");
    process::exit(2);
}

struct Args {
    artifact_root: PathBuf,
    tokenizer: PathBuf,
    prompt: String,
    raw_prompt: bool,
    max_new_tokens: usize,
    max_seq_len: usize,
    out: Option<PathBuf>,
}

fn parse_args() -> Args {
    let mut artifact_root = None;
    let mut tokenizer = None;
    let mut prompt = "Say hi.".to_owned();
    let mut raw_prompt = false;
    let mut max_new_tokens = 16usize;
    let mut max_seq_len = 128usize;
    let mut out = None;
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
            "--out" => out = Some(PathBuf::from(args.next().unwrap_or_else(|| fail(usage())))),
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
        out,
    }
}

#[cfg(not(target_os = "macos"))]
fn main() {
    fail("qwen38 native generate is Metal-only");
}

#[cfg(target_os = "macos")]
fn main() {
    let args = parse_args();
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
        "qwen38 prompt tokens={} text={rendered:?}",
        prompt_ids.len()
    );
    let mut session = Qwen38HybridDecodeSession::open(&args.artifact_root, args.max_seq_len)
        .unwrap_or_else(|e| fail(e));
    let result: Qwen38GenerateResult =
        generate_greedy(&mut session, &prompt_ids, args.max_new_tokens)
            .unwrap_or_else(|e| fail(e));
    let text = result.decode_new(&tokenizer).unwrap_or_else(|e| fail(e));
    let full = tokenizer
        .decode(&result.tokens, false)
        .unwrap_or_else(|e| fail(e));
    let gpu: Vec<u64> = result.gpu_ns.iter().copied().flatten().collect();
    let median = result.median_gpu_ns_per_token();
    println!("GENERATED_TEXT_VERBATIM: {text}");
    println!("FALLBACKS: {}", result.fallbacks);
    println!("PROMPT_LEN: {}", result.prompt_len);
    println!("NEW_TOKENS: {:?}", result.new_tokens());
    println!("GPU_NS_PER_STEP: {gpu:?}");
    println!("WAIT_NS_PER_STEP: {:?}", result.wait_ns);
    println!("MEDIAN_GPU_NS_PER_TOKEN: {median:?}");
    println!("WALL_NS: {}", result.wall_ns);
    if let Some(path) = args.out {
        if let Some(parent) = path.parent() {
            let _ = fs::create_dir_all(parent);
        }
        let body = json!({
            "lane": "qwen38-native-bringup",
            "generated_text": text,
            "full_decode": full,
            "prompt": rendered,
            "prompt_ids": prompt_ids,
            "new_token_ids": result.new_tokens(),
            "fallbacks": result.fallbacks,
            "gpu_ns_per_step": result.gpu_ns,
            "wait_ns_per_step": result.wait_ns,
            "median_gpu_ns_per_token": median,
            "wall_ns": result.wall_ns,
            "timing_label": "DIRTY_ENGINEERING",
        });
        fs::write(&path, serde_json::to_vec_pretty(&body).expect("json"))
            .unwrap_or_else(|e| fail(e));
        eprintln!("wrote {}", path.display());
    }
}

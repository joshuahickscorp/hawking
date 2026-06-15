//! Track 6 CLI surface for the spec replay-oracle. Pure CPU: `--help` needs no
//! model; the full run mmaps the GGUF tokenizer + replays an n-gram draft (no
//! Metal, no forward). Shells out to the built binary because the `dismantle`
//! crate is bin-only.
use std::process::Command;

const BIN: &str = env!("CARGO_BIN_EXE_dismantle");

#[test]
fn spec_oracle_help_lists_flags() {
    let out = Command::new(BIN)
        .args(["spec-oracle", "--help"])
        .output()
        .expect("run spec-oracle --help");
    assert!(out.status.success(), "spec-oracle --help must exit 0");
    let s = String::from_utf8_lossy(&out.stdout);
    for flag in ["--corpus", "--tokenizer-from", "--k", "--warm-frac", "--json"] {
        assert!(s.contains(flag), "help missing {flag}:\n{s}");
    }
}

#[test]
fn spec_oracle_runs_on_real_model_when_present() {
    // CPU-only end-to-end (gguf mmap -> tokenizer -> encode -> replay_grid).
    // Skipped (passes) when the model isn't checked out, so CI stays green.
    let model = concat!(env!("CARGO_MANIFEST_DIR"), "/../../models/Qwen2.5-3B-Instruct-Q4_K_M.gguf");
    if !std::path::Path::new(model).exists() {
        eprintln!("skip: model not present at {model}");
        return;
    }
    let dir = std::env::temp_dir().join("dismantle_spec_oracle_it");
    std::fs::create_dir_all(&dir).unwrap();
    let corpus = dir.join("corpus.txt");
    std::fs::write(
        &corpus,
        "the quick brown fox jumps over the lazy dog. the quick brown fox jumps over the lazy dog. \
         the quick brown fox jumps over the lazy dog. repeat repeat repeat repeat repeat the the \
         the the the the and and and and and and and.",
    )
    .unwrap();
    let out = Command::new(BIN)
        .args(["spec-oracle", "--corpus"])
        .arg(&corpus)
        .args(["--tokenizer-from", model, "--k", "4,7", "--warm-frac", "0.3", "--json"])
        .output()
        .expect("run spec-oracle on real model");
    assert!(out.status.success(), "stderr:\n{}", String::from_utf8_lossy(&out.stderr));
    let s = String::from_utf8_lossy(&out.stdout);
    assert!(s.contains("\"verdict\""), "json must carry a verdict:\n{s}");
    assert!(s.contains("\"per_k\""), "json must carry per_k rows:\n{s}");
    assert!(s.contains("\"tau\""), "json must carry tau:\n{s}");
}

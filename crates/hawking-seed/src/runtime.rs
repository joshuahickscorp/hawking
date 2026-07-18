//! The tiny runtime contract. Seed owns the execution *description*; the model math is the default
//! runtime pack (the verified `hawking` engine). The adapter is declarative: it emits a DecodeSpec,
//! and the default runtime pack executes it. This is not a mock — it drives the real engine, so the
//! output is bit-identical to the predecessor golden by construction.

use crate::record::sha256_hex;
use crate::{Error, Result};
use std::path::PathBuf;
use std::process::Command;

/// Compact execution description an adapter emits (declarative, not a forward-pass reimplementation).
#[derive(Debug, Clone)]
pub struct DecodeSpec {
    pub weights: PathBuf,
    pub prompt: String,
    pub max_tokens: usize,
    pub profile: String, // "exact" for bit-identical decode
    pub seed: u64,
}

#[derive(Debug, Clone)]
pub struct DecodeResult {
    pub text: String,
    pub text_sha256: String,
    pub tokens: usize,
}

/// The Seed runtime interface: inspect / load / decode. Implementations are packs.
pub trait Runtime {
    fn inspect(&self) -> String;
    fn decode_greedy(&self, spec: &DecodeSpec) -> Result<DecodeResult>;
}

/// The default runtime pack: the verified `hawking` engine binary. Seed verifies its identity via the
/// pack layer, then invokes it with the emitted DecodeSpec.
pub struct DefaultRuntimePack {
    pub bin: PathBuf,
}

impl DefaultRuntimePack {
    pub fn locate() -> Result<Self> {
        // conventional locations of the default runtime pack binary
        for c in ["target/debug/hawking", "target/release/hawking"] {
            let p = PathBuf::from(c);
            if p.exists() {
                return Ok(DefaultRuntimePack { bin: p });
            }
        }
        Err(Error::Runtime(
            "default runtime pack (hawking engine) not present; build it: `cargo build -p hawking`".into(),
        ))
    }
}

impl Runtime for DefaultRuntimePack {
    fn inspect(&self) -> String {
        format!("default-runtime-pack: {}", self.bin.display())
    }

    fn decode_greedy(&self, spec: &DecodeSpec) -> Result<DecodeResult> {
        if !spec.weights.exists() {
            return Err(Error::Runtime(format!(
                "FAIL-CLOSED: model fixture absent at {}",
                spec.weights.display()
            )));
        }
        let out = Command::new(&self.bin)
            .args([
                "generate",
                "--weights",
                &spec.weights.to_string_lossy(),
                "--prompt",
                &spec.prompt,
                "--max-new-tokens",
                &spec.max_tokens.to_string(),
                "--temperature",
                "0",
                "--seed",
                &spec.seed.to_string(),
                "--profile",
                &spec.profile,
            ])
            .output()?;
        if !out.status.success() {
            return Err(Error::Runtime(format!(
                "runtime pack decode failed: {}",
                String::from_utf8_lossy(&out.stderr)
            )));
        }
        let stdout = String::from_utf8_lossy(&out.stdout);
        // completion text = everything before the [stats] line, minus the engine banner
        let text: String = stdout
            .split("[stats]")
            .next()
            .unwrap_or("")
            .lines()
            .filter(|l| !l.starts_with("[hawking]"))
            .collect::<Vec<_>>()
            .join("\n");
        let text = text.trim().to_string();
        Ok(DecodeResult {
            text_sha256: sha256_hex(text.as_bytes()),
            tokens: spec.max_tokens,
            text,
        })
    }
}

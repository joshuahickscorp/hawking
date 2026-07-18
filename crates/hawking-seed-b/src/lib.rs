//! Hawking Seed — Candidate B: a self-contained runtime.
//!
//! Unlike Candidate A (which delegates model math to the 52k predecessor engine), Candidate B
//! contains its OWN compact runtime: a GGUF reader, four GGML dequantizers (Q8_0/Q5_0/Q6_K/Q4_K),
//! a GPT-2 byte-level BPE tokenizer, a small execution IR, the scalar CPU operations, and a Llama
//! forward pass — all implemented here. The only external crates are serialization (serde/serde_json),
//! hashing (sha2), the CLI (clap), an error-derive (thiserror), and the IEEE f16 numeric primitive
//! (half). No linear-algebra library, no model framework, no call into hawking-core.
//!
//! Authority (Record envelope, transition engine, Gravity policy, pack verification, evidence) is
//! reused from Candidate A — it is already the single canonical-JSON + seal engine.

// authority (reused from Candidate A: the seal/identity/control/evidence core)
pub mod evidence;
pub mod gravity;
pub mod pack;
pub mod record;
pub mod state;

// self-contained runtime (the ideological core of Candidate B)
pub mod adapter;
pub mod gguf;
pub mod ir;
pub mod model;
pub mod ops;
pub mod quant;
pub mod tokenizer;

// Gravity's Forge + Doctor fixtures, integrated with Candidate B's own linear path
pub mod doctor;
pub mod forge;

#[derive(Debug, thiserror::Error)]
pub enum Error {
    #[error("seal: {0}")]
    Seal(String),
    #[error("transition: {0}")]
    Transition(String),
    #[error("pack: {0}")]
    Pack(String),
    #[error("gravity: {0}")]
    Gravity(String),
    #[error("gguf: {0}")]
    Gguf(String),
    #[error("tokenizer: {0}")]
    Tokenizer(String),
    #[error("model: {0}")]
    Model(String),
    #[error("runtime: {0}")]
    Runtime(String),
    #[error("io: {0}")]
    Io(#[from] std::io::Error),
    #[error("json: {0}")]
    Json(#[from] serde_json::Error),
}

pub type Result<T> = std::result::Result<T, Error>;

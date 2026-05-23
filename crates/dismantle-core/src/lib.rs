#![allow(clippy::all)]
pub mod attn;
pub mod cache;
pub mod gguf;
pub mod kernel_bench;
pub mod kernels;
pub mod metal;
pub mod mixed_quant_store;
pub mod model;
pub mod moe;
pub mod profile;
pub mod quant;
pub mod quant_tier_map;
pub mod sample;
pub mod speculate;
pub mod tokenizer;
pub mod vocab_prune;

mod error;
pub use error::{Error, Result};

mod engine;
pub use engine::{
    Engine, EngineConfig, GenStats, GenerateRequest,
    SamplingParams, SpeculateMode, StopReason, StreamEvent,
};

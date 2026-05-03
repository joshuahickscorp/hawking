//! dismantle-core: Apple Silicon MoE inference.
//!
//! Three layers, exposed as modules:
//!
//! - **Runtime**: [`metal`], [`kernels`], [`quant`], [`sample`] — pure
//!   Metal glue, no model knowledge.
//! - **Model**: [`moe`], [`attn`], [`model`], [`gguf`], [`tokenizer`],
//!   [`cache`], [`speculate`] — composes runtime kernels into a
//!   forward pass.
//! - **Engine API** (this file): the [`Engine`] trait that
//!   `dismantle-serve` and `dismantle-bench` drive through.

pub mod attn;
pub mod cache;
pub mod gguf;
pub mod kernels;
pub mod metal;
pub mod model;
pub mod moe;
pub mod profile;
pub mod quant;
pub mod sample;
pub mod speculate;
pub mod tokenizer;

mod error;
pub use error::{Error, Result};

mod engine;
pub use engine::{
    Engine, EngineConfig, GenStats, GenerateRequest, SamplingParams, SpeculateMode, StopReason,
    StreamEvent,
};

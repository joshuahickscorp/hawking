//! dismantle-core: Apple Silicon MoE inference.
//!
//! The kernel/model code predates the current clippy lint set. Keep release
//! gating focused on build, parity, and smoke until a dedicated lint cleanup.

#![allow(clippy::all)]
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
pub mod kernel_bench;
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
    ActivationDtype, Engine, EngineConfig, GenStats, GenerateRequest, ResidualDtype,
    SamplingParams, SpeculateMode, StopReason, StreamEvent,
};

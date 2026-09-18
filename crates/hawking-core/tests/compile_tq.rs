//! Compile-time coverage for the optional Trellis-Quant integration sources.
//! The fast lane selects this target with `--features tq` and `cargo check`.

#[path = "qwen_tq_serve_parity.rs"]
mod qwen_tq_serve_parity;
#[path = "rwkv7_tq_bench.rs"]
mod rwkv7_tq_bench;
#[path = "rwkv7_tq_loader.rs"]
mod rwkv7_tq_loader;
#[path = "rwkv7_tq_parity.rs"]
mod rwkv7_tq_parity;
#[path = "tq_llama_probe.rs"]
mod tq_llama_probe;
#[path = "tq_output_space_quality.rs"]
mod tq_output_space_quality;
#[path = "tq_trellis_parity.rs"]
mod tq_trellis_parity;

//! Shared helpers for the Metal parity / kernel-correctness integration tests.
//!
//! Extracted to de-duplicate ~40 test files that each carried byte-identical
//! copies of these leaf utilities (`ctx`, `new_f32_buf`, `read_f32_buf`,
//! `fixed_f32`, `max_abs_diff`, `ATOL`). Behavior is identical to the inlined
//! copies — the only normalization was collapsing differing `.expect()`
//! panic-message strings on `ctx()` to a single message (no test asserts on it).
//!
//! NOTE: not `#[cfg]`-gated — it is only ever pulled in via `mod common;` from
//! the `#![cfg(target_os = "macos")]` test binaries, so it never compiles off
//! macOS. Different tolerances / by-value contexts / synthetic-input generators
//! stay local to the files that need them.
#![allow(dead_code)]

use hawking_core::metal::{MetalContext, PinnedBuffer};
use once_cell::sync::Lazy;
use rand::Rng;
use rand_pcg::Pcg64Mcg;

/// Process-wide lazily-initialized Metal context (one device per test binary).
pub fn ctx() -> &'static MetalContext {
    static CTX: Lazy<MetalContext> =
        Lazy::new(|| MetalContext::new().expect("Metal device required"));
    &CTX
}

/// Deterministic pseudo-random `f32` vector in `[-1, 1)` from a fixed seed.
pub fn fixed_f32(n: usize, seed: u64) -> Vec<f32> {
    let mut rng = Pcg64Mcg::new(seed as u128);
    (0..n).map(|_| rng.gen_range(-1.0_f32..1.0_f32)).collect()
}

/// Upload an `f32` slice into a pinned Metal buffer.
pub fn new_f32_buf(ctx: &MetalContext, data: &[f32]) -> PinnedBuffer {
    ctx.new_buffer_with_bytes(bytemuck::cast_slice(data))
}

/// Read `n` `f32`s back out of a pinned Metal buffer.
pub fn read_f32_buf(buf: &PinnedBuffer, n: usize) -> Vec<f32> {
    let ptr = buf.contents() as *const f32;
    unsafe { std::slice::from_raw_parts(ptr, n) }.to_vec()
}

/// Maximum absolute element-wise difference between two equal-length slices.
pub fn max_abs_diff(a: &[f32], b: &[f32]) -> f32 {
    a.iter()
        .zip(b.iter())
        .map(|(&x, &y)| (x - y).abs())
        .fold(0.0_f32, f32::max)
}

/// Standard fp16 parity tolerance (Metal kernel vs CPU reference).
pub const ATOL: f32 = 1e-3;

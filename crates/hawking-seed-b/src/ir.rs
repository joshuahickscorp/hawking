//! The compact execution IR. A `Plan` is a flat, ordered list of typed `Op`s over a small register
//! file. Each op encodes its inputs, output, parameters, and — for weight ops — the tensor identity,
//! dimensions, and quantization format. The adapter emits a Plan from model metadata; the runtime
//! (model.rs) interprets it. This is sufficient for one real SmolLM/Llama decode path and replaces a
//! large architecture-specific hand-written `forward()`.

use serde::Serialize;

/// The register file. Small, fixed set of activation buffers the ops read/write. `X` is the residual
/// stream; the rest are scratch. Backing buffers live in the runtime, indexed by `Reg as usize`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
pub enum Reg {
    X = 0,      // residual stream (hidden)
    Xn = 1,     // normalized scratch
    Q = 2,      // query projection
    K = 3,      // key projection
    V = 4,      // value projection
    Attn = 5,   // attention output
    G = 6,      // ffn gate
    U = 7,      // ffn up
    A = 8,      // ffn activated (silu(g)*u)
    Logits = 9, // vocab logits
}
pub const N_REGS: usize = 10;

/// A weight reference: identity + shape + source quantization format (the IR records the format even
/// though the runtime executes on the dequantized f32/f16 reference path).
#[derive(Debug, Clone, Serialize)]
pub struct TensorRef {
    pub name: String,
    pub rows: usize, // out_features
    pub cols: usize, // in_features
    pub quant: String,
}

/// One IR operation. Ordered execution; no control flow (the plan is fully unrolled across layers).
#[derive(Debug, Clone, Serialize)]
pub enum Op {
    /// hidden = embedding row for the runtime's current token (f16 weight).
    Embed { out: Reg, weight: TensorRef },
    /// dst = rmsnorm(src, weight, eps)  [f64 sum-of-squares]
    RmsNorm { src: Reg, dst: Reg, weight: TensorRef, eps: f32, n: usize },
    /// dst = weight @ src   [out-major f32 GEMV]
    Linear { src: Reg, dst: Reg, weight: TensorRef },
    /// in-place NeoX RoPE over `reg` for `n_heads` heads of `head_dim`, base theta, runtime position.
    Rope { reg: Reg, n_heads: usize, head_dim: usize, base: f32 },
    /// append the current K/V registers into layer `layer`'s cache at the runtime position.
    KvWrite { layer: usize, k: Reg, v: Reg },
    /// out = attention(q, cached K/V for `layer`) with GQA + causal masking to the current position.
    Attention { q: Reg, out: Reg, layer: usize, n_heads: usize, n_kv_heads: usize, head_dim: usize },
    /// dst += add   [residual]
    Residual { dst: Reg, add: Reg },
    /// dst = silu(gate) * up
    SiluMul { gate: Reg, up: Reg, dst: Reg },
    /// dst = tied_embedding^T @ src   [f16 weight, vocab logits]
    Logits { src: Reg, dst: Reg, weight: TensorRef },
    /// greedy argmax over `src` -> runtime.next_token (executed only in decode, skipped in prefill).
    Sample { src: Reg },
}

/// A full forward plan plus the config the runtime needs at execution time.
#[derive(Debug, Clone, Serialize)]
pub struct Plan {
    pub ops: Vec<Op>,
    pub n_layers: usize,
    pub hidden: usize,
    pub vocab: usize,
}

impl Plan {
    /// Count of ops by kind — a small structural summary for `inspect`.
    pub fn summary(&self) -> String {
        let mut linear = 0;
        let mut norm = 0;
        let mut attn = 0;
        for op in &self.ops {
            match op {
                Op::Linear { .. } | Op::Logits { .. } => linear += 1,
                Op::RmsNorm { .. } => norm += 1,
                Op::Attention { .. } => attn += 1,
                _ => {}
            }
        }
        format!("{} ops: {linear} linear, {norm} rmsnorm, {attn} attention, {} layers", self.ops.len(), self.n_layers)
    }
}

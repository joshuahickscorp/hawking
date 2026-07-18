//! The model adapter: read Llama metadata + tensor shapes from the GGUF and emit an execution `Plan`.
//! This is primarily data + tensor mapping — the adapter knows the Llama block shape, the runtime
//! knows how to execute each op. Supports exactly one parent: SmolLM-135M / Llama.

use crate::gguf::{GgmlType, GgufFile};
use crate::ir::{Op, Plan, Reg, TensorRef};
use crate::{Error, Result};

#[derive(Debug, Clone)]
pub struct LlamaConfig {
    pub n_layers: usize,
    pub hidden: usize,
    pub n_ff: usize,
    pub n_heads: usize,
    pub n_kv_heads: usize,
    pub head_dim: usize,
    pub rope_base: f32,
    pub rms_eps: f32,
    pub vocab: usize,
    pub context: usize,
}

impl LlamaConfig {
    pub fn from_gguf(g: &GgufFile) -> Result<Self> {
        let arch = g.meta_str("general.architecture").unwrap_or("llama");
        if arch != "llama" {
            return Err(Error::Model(format!("adapter supports arch=llama, got {arch}")));
        }
        let n_heads = g.meta_u32("llama.attention.head_count")? as usize;
        let n_kv_heads = g.meta_u32("llama.attention.head_count_kv")? as usize;
        let hidden = g.meta_u32("llama.embedding_length")? as usize;
        let head_dim = g
            .meta_u32("llama.attention.key_length")
            .map(|v| v as usize)
            .unwrap_or(hidden / n_heads);
        Ok(LlamaConfig {
            n_layers: g.meta_u32("llama.block_count")? as usize,
            hidden,
            n_ff: g.meta_u32("llama.feed_forward_length")? as usize,
            n_heads,
            n_kv_heads,
            head_dim,
            rope_base: g.meta_f32("llama.rope.freq_base").unwrap_or(10000.0),
            rms_eps: g.meta_f32("llama.attention.layer_norm_rms_epsilon").unwrap_or(1e-5),
            vocab: g.meta_u32("llama.vocab_size").unwrap_or(0) as usize,
            context: g.meta_u32("llama.context_length").unwrap_or(2048) as usize,
        })
    }
    pub fn q_dim(&self) -> usize {
        self.n_heads * self.head_dim
    }
    pub fn kv_dim(&self) -> usize {
        self.n_kv_heads * self.head_dim
    }
}

fn quant_name(t: GgmlType) -> String {
    format!("{t:?}")
}

/// A 2-D weight used as `out = W @ x`: GGUF dims = [in, out] so rows(out)=dims[1], cols(in)=dims[0].
fn tref_2d(g: &GgufFile, name: &str) -> Result<TensorRef> {
    let t = g.tensor(name)?;
    let (cols, rows) = match t.dims.as_slice() {
        [c, r] => (*c as usize, *r as usize),
        _ => return Err(Error::Model(format!("{name}: expected 2-D weight, got {:?}", t.dims))),
    };
    Ok(TensorRef { name: name.into(), rows, cols, quant: quant_name(t.dtype) })
}

/// A 1-D norm weight (length = hidden).
fn tref_1d(g: &GgufFile, name: &str) -> Result<TensorRef> {
    let t = g.tensor(name)?;
    let n = t.dims.first().copied().unwrap_or(0) as usize;
    Ok(TensorRef { name: name.into(), rows: n, cols: 1, quant: quant_name(t.dtype) })
}

/// Emit the full unrolled forward plan for the configured Llama model.
pub fn build_plan(g: &GgufFile, cfg: &LlamaConfig) -> Result<Plan> {
    let mut ops = Vec::new();
    let embd = tref_2d(g, "token_embd.weight")?; // rows=vocab, cols=hidden

    ops.push(Op::Embed { out: Reg::X, weight: embd.clone() });

    for l in 0..cfg.n_layers {
        let p = |s: &str| format!("blk.{l}.{s}");
        // attention block (pre-norm)
        ops.push(Op::RmsNorm {
            src: Reg::X,
            dst: Reg::Xn,
            weight: tref_1d(g, &p("attn_norm.weight"))?,
            eps: cfg.rms_eps,
            n: cfg.hidden,
        });
        ops.push(Op::Linear { src: Reg::Xn, dst: Reg::Q, weight: tref_2d(g, &p("attn_q.weight"))? });
        ops.push(Op::Linear { src: Reg::Xn, dst: Reg::K, weight: tref_2d(g, &p("attn_k.weight"))? });
        ops.push(Op::Linear { src: Reg::Xn, dst: Reg::V, weight: tref_2d(g, &p("attn_v.weight"))? });
        ops.push(Op::Rope { reg: Reg::Q, n_heads: cfg.n_heads, head_dim: cfg.head_dim, base: cfg.rope_base });
        ops.push(Op::Rope { reg: Reg::K, n_heads: cfg.n_kv_heads, head_dim: cfg.head_dim, base: cfg.rope_base });
        ops.push(Op::KvWrite { layer: l, k: Reg::K, v: Reg::V });
        ops.push(Op::Attention {
            q: Reg::Q,
            out: Reg::Attn,
            layer: l,
            n_heads: cfg.n_heads,
            n_kv_heads: cfg.n_kv_heads,
            head_dim: cfg.head_dim,
        });
        ops.push(Op::Linear { src: Reg::Attn, dst: Reg::Xn, weight: tref_2d(g, &p("attn_output.weight"))? });
        ops.push(Op::Residual { dst: Reg::X, add: Reg::Xn });
        // ffn block (pre-norm, SwiGLU)
        ops.push(Op::RmsNorm {
            src: Reg::X,
            dst: Reg::Xn,
            weight: tref_1d(g, &p("ffn_norm.weight"))?,
            eps: cfg.rms_eps,
            n: cfg.hidden,
        });
        ops.push(Op::Linear { src: Reg::Xn, dst: Reg::G, weight: tref_2d(g, &p("ffn_gate.weight"))? });
        ops.push(Op::Linear { src: Reg::Xn, dst: Reg::U, weight: tref_2d(g, &p("ffn_up.weight"))? });
        ops.push(Op::SiluMul { gate: Reg::G, up: Reg::U, dst: Reg::A });
        ops.push(Op::Linear { src: Reg::A, dst: Reg::Xn, weight: tref_2d(g, &p("ffn_down.weight"))? });
        ops.push(Op::Residual { dst: Reg::X, add: Reg::Xn });
    }

    ops.push(Op::RmsNorm {
        src: Reg::X,
        dst: Reg::Xn,
        weight: tref_1d(g, "output_norm.weight")?,
        eps: cfg.rms_eps,
        n: cfg.hidden,
    });
    ops.push(Op::Logits { src: Reg::Xn, dst: Reg::Logits, weight: embd }); // tied
    ops.push(Op::Sample { src: Reg::Logits });

    Ok(Plan { ops, n_layers: cfg.n_layers, hidden: cfg.hidden, vocab: cfg.vocab })
}

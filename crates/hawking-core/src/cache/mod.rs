//! KV cache — both in-memory (every phase) and on-disk
//! cross-session (wedge 5, Phase 5).
//!
//! The on-disk variant mmaps a previously-computed KV cache for a
//! known system prompt prefix. Cache key includes model hash,
//! tokenizer hash, and prompt content hash. Mismatch falls back to a
//! cold prefill. Drops cold-start TTFT by orders of magnitude on
//! system-prompt-heavy workloads.

pub mod prefill_disk;

use crate::Result;

/// Per-layer KV cache. Stored as fp16 in flight; the model layer
/// converts on append/read. Logical shape per layer:
///   keys:   (max_seq, n_kv_heads, head_dim)
///   values: (max_seq, n_kv_heads, head_dim)
///
/// For DeepSeek MLA, "head_dim" is the latent dim and `n_kv_heads`
/// collapses to 1 — the model layer reshapes appropriately.
#[derive(Debug)]
pub struct KvCache {
    pub max_seq: usize,
    pub n_layers: usize,
    pub n_kv_heads: usize,
    pub head_dim: usize,
    /// keys[layer] is a (max_seq * n_kv_heads * head_dim) flat fp32
    /// buffer; using fp32 in the reference path keeps numerics simple.
    /// Phase 1+ converts to fp16 via the Metal-side append kernel.
    pub keys: Vec<Vec<f32>>,
    pub values: Vec<Vec<f32>>,
    /// Current filled length — same across layers because we always
    /// run the model layer-synchronously.
    pub seq_len: usize,
}

impl KvCache {
    pub fn new(n_layers: usize, max_seq: usize, n_kv_heads: usize, head_dim: usize) -> Self {
        let per_layer = max_seq * n_kv_heads * head_dim;
        Self {
            max_seq,
            n_layers,
            n_kv_heads,
            head_dim,
            keys: (0..n_layers).map(|_| vec![0.0f32; per_layer]).collect(),
            values: (0..n_layers).map(|_| vec![0.0f32; per_layer]).collect(),
            seq_len: 0,
        }
    }

    /// Append one token's K and V into every layer's cache. Layer
    /// arrays must be (n_kv_heads * head_dim) each, in head-major order.
    pub fn append(&mut self, k_per_layer: &[Vec<f32>], v_per_layer: &[Vec<f32>]) -> Result<()> {
        if self.seq_len >= self.max_seq {
            return Err(crate::Error::Model(format!(
                "kv cache full at {}",
                self.max_seq
            )));
        }
        let stride = self.n_kv_heads * self.head_dim;
        for layer in 0..self.n_layers {
            let off = self.seq_len * stride;
            self.keys[layer][off..off + stride].copy_from_slice(&k_per_layer[layer]);
            self.values[layer][off..off + stride].copy_from_slice(&v_per_layer[layer]);
        }
        self.seq_len += 1;
        Ok(())
    }

    /// Slice the keys for a given layer up to current seq_len.
    pub fn keys_for(&self, layer: usize) -> &[f32] {
        let stride = self.n_kv_heads * self.head_dim;
        &self.keys[layer][..self.seq_len * stride]
    }

    pub fn values_for(&self, layer: usize) -> &[f32] {
        let stride = self.n_kv_heads * self.head_dim;
        &self.values[layer][..self.seq_len * stride]
    }

    pub fn reset(&mut self) {
        self.seq_len = 0;
    }
}

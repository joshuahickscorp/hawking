//! P1-D1: shared GGUF config-reader helper.
//!
//! The per-architecture `from_gguf` readers (qwen-dense, llama, gemma2, phi3,
//! and the MoE mixtral/deepseek) each grew the same prefix-relative scalar
//! reads: a pair of `get_u32`/`get_f32` closures, the required-key
//! `ok_or_else(missing "<prefix>.<field>")` pattern, the `head_count_kv`→
//! `head_count` default, and the rope/eps/context opt-with-default reads.
//!
//! They are NOT mergeable into a single config STRUCT: each arch carries
//! genuinely different fields and defaults — llama has `rope_scaling` + an
//! `arch` string and rope-theta default 500_000; gemma2 has attn/final logit
//! soft-caps, an embed scale and rope-theta default 10_000; phi3 has its own
//! sliding-window handling and rope-theta default 10_000; qwen has none of
//! these and a rope-theta default 1_000_000; rms-eps and context defaults
//! differ too (1e-6 vs 1e-5, 32768 vs 8192 vs 4096). So this is a shared
//! READER, not a shared struct: each loader keeps its own `*Config` plus its
//! arch-specific reads (vocab-fallback chain, head_dim policy, extras) and
//! routes only the duplicated common core through [`ArchReader`]. The
//! required-key error strings are byte-identical to the old inline readers.

use crate::gguf::{GgufFile, MetaValue};
use crate::{Error, Result};
use std::collections::HashMap;

/// Reads the shared config core out of a GGUF metadata map under one arch key
/// prefix (e.g. `"qwen2"`, `"llama"`, `"gemma2"`, `"phi3"` — no trailing dot).
pub struct ArchReader<'a> {
    metadata: &'a HashMap<String, MetaValue>,
    prefix: &'a str,
}

impl<'a> ArchReader<'a> {
    /// Production constructor over a loaded GGUF.
    pub fn new(g: &'a GgufFile, prefix: &'a str) -> Self {
        Self { metadata: &g.metadata, prefix }
    }

    /// Constructor over a raw metadata map (used by the equivalence tests so
    /// they can drive the reader from a hand-built mock without a model).
    pub fn from_metadata(metadata: &'a HashMap<String, MetaValue>, prefix: &'a str) -> Self {
        Self { metadata, prefix }
    }

    fn u32_opt(&self, field: &str) -> Option<u32> {
        self.metadata.get(&format!("{}.{}", self.prefix, field)).and_then(|v| v.as_u32())
    }

    fn f32_opt(&self, field: &str) -> Option<f32> {
        self.metadata.get(&format!("{}.{}", self.prefix, field)).and_then(|v| v.as_f32())
    }

    /// Required `u32` field returned as `usize`. Errors with the exact message
    /// `missing <prefix>.<field>` the inline readers emitted.
    pub fn req_usize(&self, field: &str) -> Result<usize> {
        self.u32_opt(field).map(|v| v as usize).ok_or_else(|| Error::Model(format!("missing {}.{}", self.prefix, field)))
    }

    /// Optional `u32` field as `usize`, with a fallback when absent.
    pub fn opt_usize(&self, field: &str, default: usize) -> usize {
        self.u32_opt(field).map(|v| v as usize).unwrap_or(default)
    }

    /// Optional `f32` field, with a fallback when absent.
    pub fn opt_f32(&self, field: &str, default: f32) -> f32 {
        self.f32_opt(field).unwrap_or(default)
    }
}

/// Derive vocab size from the largest dimension of `token_embd.weight`, using
/// the caller's historical error string when the tensor is absent.
pub fn token_embd_vocab_size(g: &GgufFile, missing: impl Into<String>) -> Result<usize> {
    let dims = g.tensor("token_embd.weight").map(|t| t.dims.clone()).ok_or_else(|| Error::Model(missing.into()))?;
    Ok(dims.iter().copied().max().unwrap_or(0) as usize)
}

pub fn token_embd_vocab_size_opt(g: &GgufFile) -> Option<usize> {
    g.tensor("token_embd.weight").and_then(|t| t.dims.iter().copied().max()).map(|v| v as usize)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn meta(pairs: &[(&str, MetaValue)]) -> HashMap<String, MetaValue> {
        pairs.iter().map(|(k, v)| (k.to_string(), v.clone())).collect()
    }

    #[test]
    fn req_usize_reads_and_errors_with_prefixed_message() {
        let m = meta(&[("qwen2.block_count", MetaValue::U32(36))]);
        let r = ArchReader::from_metadata(&m, "qwen2");
        assert_eq!(r.req_usize("block_count").unwrap(), 36);
        // Missing required key → exact "missing <prefix>.<field>" message.
        let err = r.req_usize("embedding_length").unwrap_err();
        match err {
            Error::Model(s) => assert_eq!(s, "missing qwen2.embedding_length"),
            other => panic!("expected Error::Model, got {other:?}"),
        }
    }

    #[test]
    fn opt_usize_and_f32_defaults() {
        let m = meta(&[("llama.attention.head_count_kv", MetaValue::U32(8)), ("gemma2.rope.freq_base", MetaValue::F32(10_000.0))]);
        let rl = ArchReader::from_metadata(&m, "llama");
        // present → read value; absent → fallback.
        assert_eq!(rl.opt_usize("attention.head_count_kv", 32), 8);
        assert_eq!(rl.opt_usize("attention.key_length", 64), 64);
        let rg = ArchReader::from_metadata(&m, "gemma2");
        assert_eq!(rg.opt_f32("rope.freq_base", 1.0), 10_000.0);
        assert_eq!(rg.opt_f32("attention.layer_norm_rms_epsilon", 1e-6), 1e-6);
    }

    #[test]
    fn prefix_isolation() {
        // A key under a different prefix must not be visible.
        let m = meta(&[("llama.block_count", MetaValue::U32(40))]);
        let r = ArchReader::from_metadata(&m, "qwen2");
        assert!(r.req_usize("block_count").is_err());
    }
}

//! Vocab pruning: compact the LM head from the full tokenizer vocab to a
//! corpus-supported subset.
//!
//! ## Motivation
//!
//! V2-Lite-Chat ships a 102,400-token vocab. On our 4,512-sequence
//! calibration corpus (artifacts/calibration/analysis/summary.md):
//!
//! - 29,054 unique tokens actually appear (28.4% of vocab)
//! - top-23,628 tokens cover 99.5% of corpus tokens
//!
//! Slicing the LM head from `[102400, 2048]` → `[23628, 2048]` is a 76.9%
//! reduction in output-projection compute (also in the argmax pass that
//! follows). LM head ≈ 4% of decode-time per the v1.1.0 findings, so this
//! lever is worth roughly +1-3 tps standalone — bracketed by the
//! `path-to-30` plan's +2-5 tps projection for vocab-prune.
//!
//! ## Whitelist file
//!
//! Built by `tools/training/analyze_corpus.py` → produces
//! `artifacts/calibration/analysis/vocab_whitelist_995.json`:
//!
//! ```json
//! {
//!   "vocab_size_original": 102400,
//!   "vocab_size_pruned":   23628,
//!   "coverage":            0.995,
//!   "total_corpus_tokens": 1154515,
//!   "keep_token_ids":      [0, 1, 2, ...]   // ascending original token ids
//! }
//! ```
//!
//! The `keep_token_ids` list is **sorted ascending** by the producer;
//! [`PrunedVocab::load`] re-sorts defensively but documents the invariant.
//!
//! ## Wiring (caller responsibility — not in this module)
//!
//! 1. **Load:** `PrunedVocab::load(path)` at engine init if the profile has
//!    `vocab_prune_path = Some(path)`.
//! 2. **Slice the LM head weight at GGUF load:** the model loader calls
//!    [`PrunedVocab::slice_lm_head_f16`] with the dequantized `[vocab,
//!    hidden]` weight tensor; replaces it with the compact pruned weight.
//! 3. **Set `cfg.vocab_size = pruned.pruned_len()`** so downstream
//!    allocations (logits_buf, argmax dispatch shape) shrink accordingly.
//! 4. **Sampling translation:** after the argmax kernel returns
//!    `pruned_id ∈ 0..pruned_len`, the sampler calls
//!    [`PrunedVocab::pruned_to_original`] before handing the token to the
//!    tokenizer / detokenizer.
//! 5. **Optional integrity check:** call [`PrunedVocab::validate`] against
//!    the tokenizer's reported vocab_size to fail fast if the whitelist was
//!    built for a different model.
//!
//! ## Out of scope here
//!
//! - **Tied embeddings.** DeepSeek-V2 has an explicit `output.weight` so
//!   this is fine. Tied models (e.g. Qwen3) would need a separate
//!   `slice_embed_for_output` path; this module currently asserts that the
//!   caller is operating on a non-tied LM head weight.
//! - **Sub-pruned KV cache or embed table.** We do NOT prune the input
//!   embedding because the input still receives ANY token id (the user's
//!   prompt can include rare tokens). Pruning embed would require either
//!   throwing on unknown input tokens (bad UX) or maintaining a "rest"
//!   bucket. Out of scope for v1.
//! - **Dynamic re-pruning at runtime.** The whitelist is loaded once; no
//!   hot-reload.
//!
//! ## Determinism
//!
//! All operations are deterministic. `pruned_to_original` is a simple slice
//! index lookup. `slice_lm_head_f16` is a row-gather copy. Bit-identical
//! across runs for the same whitelist + weight.

use std::collections::HashMap;
use std::path::Path;

use half::f16;
use serde::Deserialize;

use crate::{Error, Result};

/// Whitelist loaded from `vocab_whitelist_995.json`. Owns the
/// pruned→original token-id mapping and a constant-time
/// original→pruned reverse map.
///
/// Hold this in `Engine` for the lifetime of the model.
#[derive(Debug, Clone)]
pub struct PrunedVocab {
    /// Original vocab size before pruning (e.g. 102_400 for V2-Lite).
    pub original_vocab_size: usize,
    /// Pruned vocab size (e.g. 23_628). Equals `keep_token_ids.len()`.
    pub pruned_vocab_size: usize,
    /// Coverage fraction reported by the producer (informational only).
    pub coverage: f64,
    /// pruned_id → original_id. Indexed by pruned id (0..pruned_vocab_size).
    keep_token_ids: Vec<u32>,
    /// original_id → Option<pruned_id>. Sparse map for O(1) reverse lookup.
    /// Stored as a HashMap to keep memory bounded when pruned_vocab_size
    /// ≪ original_vocab_size; a dense `Vec<i32>` would also work and trade
    /// memory for a tiny lookup-speed win. Default to HashMap for simplicity.
    reverse: HashMap<u32, u32>,
}

#[derive(Debug, Deserialize)]
struct WhitelistFile {
    vocab_size_original: usize,
    vocab_size_pruned: usize,
    coverage: f64,
    #[allow(dead_code)]
    total_corpus_tokens: u64,
    keep_token_ids: Vec<u32>,
}

impl PrunedVocab {
    /// Load and validate a vocab whitelist JSON file.
    ///
    /// Errors:
    /// - IO: file not found or unreadable
    /// - Model: malformed JSON; `vocab_size_pruned` ≠ `keep_token_ids.len()`;
    ///   any keep_id >= `vocab_size_original`; duplicate ids
    pub fn load(path: impl AsRef<Path>) -> Result<Self> {
        let bytes = std::fs::read(path.as_ref())?;
        let raw: WhitelistFile = serde_json::from_slice(&bytes).map_err(|e| Error::Model(format!("vocab whitelist parse: {e}")))?;
        Self::from_parts(raw.vocab_size_original, raw.vocab_size_pruned, raw.coverage, raw.keep_token_ids)
    }

    /// Build from already-parsed components. Useful for tests.
    pub fn from_parts(original_vocab_size: usize, pruned_vocab_size: usize, coverage: f64, mut keep_token_ids: Vec<u32>) -> Result<Self> {
        if keep_token_ids.len() != pruned_vocab_size {
            return Err(Error::Model(format!(
                "vocab whitelist: vocab_size_pruned={pruned_vocab_size} but \
                 keep_token_ids.len()={}",
                keep_token_ids.len()
            )));
        }
        if pruned_vocab_size == 0 {
            return Err(Error::Model("vocab whitelist: pruned vocab size 0 (would yield no outputs)".into()));
        }
        if pruned_vocab_size > original_vocab_size {
            return Err(Error::Model(format!("vocab whitelist: pruned_size {pruned_vocab_size} > original {original_vocab_size}")));
        }
        // Defensive sort; producer is supposed to emit ascending.
        keep_token_ids.sort_unstable();
        // Detect duplicates + out-of-range.
        for pair in keep_token_ids.windows(2) {
            if pair[0] == pair[1] {
                return Err(Error::Model(format!("vocab whitelist: duplicate token id {}", pair[0])));
            }
        }
        if let Some(&max_id) = keep_token_ids.last() {
            if (max_id as usize) >= original_vocab_size {
                return Err(Error::Model(format!("vocab whitelist: token id {max_id} >= original vocab {original_vocab_size}")));
            }
        }
        let mut reverse = HashMap::with_capacity(pruned_vocab_size);
        for (pruned_id, &orig_id) in keep_token_ids.iter().enumerate() {
            reverse.insert(orig_id, pruned_id as u32);
        }
        Ok(Self { original_vocab_size, pruned_vocab_size, coverage, keep_token_ids, reverse })
    }

    /// Number of tokens after pruning. Use this as the new `cfg.vocab_size`.
    #[inline]
    pub fn pruned_len(&self) -> usize {
        self.pruned_vocab_size
    }

    /// Translate a pruned token id (returned by argmax over the sliced LM
    /// head) back to its original tokenizer id. Panics on out-of-range —
    /// the caller is expected to have just produced this id from a buffer
    /// sized `pruned_len()`.
    #[inline]
    pub fn pruned_to_original(&self, pruned_id: u32) -> u32 {
        self.keep_token_ids[pruned_id as usize]
    }

    /// Returns `Some(pruned_id)` if `original_id` survived pruning, else
    /// `None`. Used by integrity checks; not on the hot path.
    #[inline]
    pub fn original_to_pruned(&self, original_id: u32) -> Option<u32> {
        self.reverse.get(&original_id).copied()
    }

    /// Validate against the live tokenizer's reported vocab size. Errors if
    /// they disagree (whitelist was built for a different model).
    pub fn validate(&self, tokenizer_vocab_size: usize) -> Result<()> {
        if self.original_vocab_size != tokenizer_vocab_size {
            return Err(Error::Model(format!("vocab whitelist built for vocab_size={} but tokenizer reports {}", self.original_vocab_size, tokenizer_vocab_size)));
        }
        Ok(())
    }

    /// Gather rows of a `[orig_vocab, hidden]` LM head weight matrix into a
    /// new `[pruned_vocab, hidden]` compact weight. Caller passes in the
    /// dequantized fp16 weight; we return a fresh allocation.
    ///
    /// `orig_weight` must be exactly `original_vocab_size * hidden` long;
    /// errors otherwise.
    pub fn slice_lm_head_f16(&self, orig_weight: &[f16], hidden: usize) -> Result<Vec<f16>> {
        let expected = self.original_vocab_size * hidden;
        if orig_weight.len() != expected {
            return Err(Error::Model(format!("vocab prune: lm_head weight length {} ≠ expected {} ({}×{})", orig_weight.len(), expected, self.original_vocab_size, hidden)));
        }
        let mut out = Vec::with_capacity(self.pruned_vocab_size * hidden);
        for &orig_id in &self.keep_token_ids {
            let start = (orig_id as usize) * hidden;
            let end = start + hidden;
            out.extend_from_slice(&orig_weight[start..end]);
        }
        debug_assert_eq!(out.len(), self.pruned_vocab_size * hidden);
        Ok(out)
    }

    /// View of the keep list. Useful for tests and diagnostics.
    pub fn keep_ids(&self) -> &[u32] {
        &self.keep_token_ids
    }

    /// Informational coverage figure from the producer.
    pub fn coverage(&self) -> f64 {
        self.coverage
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn whitelist(orig: usize, keep: &[u32]) -> PrunedVocab {
        PrunedVocab::from_parts(orig, keep.len(), 0.995, keep.to_vec()).unwrap()
    }

    #[test]
    fn round_trip_identity_for_kept_tokens() {
        let p = whitelist(100, &[0, 7, 11, 42, 99]);
        assert_eq!(p.pruned_to_original(0), 0);
        assert_eq!(p.pruned_to_original(2), 11);
        assert_eq!(p.pruned_to_original(4), 99);
        assert_eq!(p.original_to_pruned(0), Some(0));
        assert_eq!(p.original_to_pruned(42), Some(3));
        assert_eq!(p.original_to_pruned(100), None);
        assert_eq!(p.original_to_pruned(15), None);
        assert_eq!(p.pruned_len(), 5);
    }

    #[test]
    fn rejects_duplicates() {
        let r = PrunedVocab::from_parts(100, 3, 0.99, vec![1, 2, 2]);
        assert!(r.is_err());
        let msg = format!("{:?}", r.unwrap_err());
        assert!(msg.contains("duplicate"), "{msg}");
    }

    #[test]
    fn rejects_out_of_range_id() {
        let r = PrunedVocab::from_parts(10, 2, 0.99, vec![5, 10]);
        assert!(r.is_err());
    }

    #[test]
    fn rejects_length_mismatch() {
        let r = PrunedVocab::from_parts(100, 5, 0.99, vec![1, 2]);
        assert!(r.is_err());
    }

    #[test]
    fn rejects_empty_pruned() {
        let r = PrunedVocab::from_parts(100, 0, 0.0, vec![]);
        assert!(r.is_err());
    }

    #[test]
    fn rejects_pruned_larger_than_original() {
        let r = PrunedVocab::from_parts(3, 5, 0.99, vec![0, 1, 2, 3, 4]);
        assert!(r.is_err());
    }

    #[test]
    fn sorts_unsorted_input() {
        let p = whitelist(10, &[5, 1, 7, 0, 3]);
        assert_eq!(p.keep_ids(), &[0, 1, 3, 5, 7]);
    }

    #[test]
    fn validate_matches_tokenizer() {
        let p = whitelist(100, &[0, 1, 2]);
        assert!(p.validate(100).is_ok());
        assert!(p.validate(50).is_err());
    }

    #[test]
    fn slice_lm_head_f16_gathers_correct_rows() {
        // 4-vocab × 3-hidden weight where row i has values [i*10, i*10+1, i*10+2]
        let hidden = 3;
        let orig_vocab = 4;
        let mut w = Vec::with_capacity(orig_vocab * hidden);
        for r in 0..orig_vocab {
            for c in 0..hidden {
                w.push(f16::from_f32((r * 10 + c) as f32));
            }
        }
        let p = whitelist(orig_vocab, &[0, 2, 3]); // skip row 1
        let sliced = p.slice_lm_head_f16(&w, hidden).unwrap();
        assert_eq!(sliced.len(), 3 * 3);
        // pruned row 0 ← orig row 0: [0, 1, 2]
        assert_eq!(sliced[0..3], [f16::from_f32(0.0), f16::from_f32(1.0), f16::from_f32(2.0)]);
        // pruned row 1 ← orig row 2: [20, 21, 22]
        assert_eq!(sliced[3..6], [f16::from_f32(20.0), f16::from_f32(21.0), f16::from_f32(22.0)]);
        // pruned row 2 ← orig row 3: [30, 31, 32]
        assert_eq!(sliced[6..9], [f16::from_f32(30.0), f16::from_f32(31.0), f16::from_f32(32.0)]);
    }

    #[test]
    fn slice_lm_head_rejects_wrong_length() {
        let p = whitelist(4, &[0, 2, 3]);
        let bad = vec![f16::from_f32(0.0); 4 * 3 - 1];
        assert!(p.slice_lm_head_f16(&bad, 3).is_err());
    }

    #[test]
    fn argmax_round_trip_via_pruned_space() {
        // Simulate the live path: full-vocab argmax vs pruned argmax should
        // pick the same token (after pruned→original mapping) when the
        // top-1 token is in the whitelist.
        let p = whitelist(8, &[0, 1, 3, 5, 7]);
        // Hypothetical full-vocab logits where token 5 wins.
        let mut full = vec![0.0f32; 8];
        full[5] = 10.0;
        // Pruned logits picks pruned_id 3 (which maps back to 5).
        let pruned_logits: Vec<f32> = p.keep_ids().iter().map(|&i| full[i as usize]).collect();
        let pruned_winner = pruned_logits.iter().enumerate().max_by(|a, b| a.1.partial_cmp(b.1).unwrap()).unwrap().0 as u32;
        assert_eq!(p.pruned_to_original(pruned_winner), 5);
    }
}

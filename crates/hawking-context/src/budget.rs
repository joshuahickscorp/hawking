//! Token budgeting and tokenizer-accurate counting (bible §4.2).
//!
//! The budget is **reserve-then-fill** (bible §4.2.3, F1): the system,
//! response, and scratchpad regions are carved out *before* any candidate
//! competes, so an overflow can never eat the response budget. Counting goes
//! through a real `tokenizers` tokenizer when one is available, falling back
//! to a `chars/4` heuristic otherwise (bible §4.2 "tokenizer-accurate counting,
//! chars/4 fallback").

use parking_lot::RwLock;
use serde::{Deserialize, Serialize};
use std::sync::Arc;
use tokenizers::Tokenizer;

/// The window budget for one compile, with hard per-region reservations.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct TokenBudget {
    /// Total tokens the model window can hold (effective ctx_len).
    pub max_input_tokens: usize,
    /// Tokens reserved for the model to *generate* (bible F1 — carved first).
    pub reserve_output_tokens: usize,
    /// A floor on the window we will actually fill (lets a profile cap a huge
    /// native window for a tight task).
    pub hard_limit_tokens: usize,
}

impl TokenBudget {
    /// Input tokens available *after* the output reservation, clamped to the
    /// hard limit. This is the pool the packer fills.
    pub fn available_input(&self) -> usize {
        self.max_input_tokens
            .saturating_sub(self.reserve_output_tokens)
            .min(self.hard_limit_tokens)
    }

    /// Reserve a percentage of the available input as a named region.
    pub fn reserve_pct(&self, pct: f32) -> usize {
        ((self.available_input() as f32) * pct.clamp(0.0, 1.0)).floor() as usize
    }
}

/// A named, budgeted region of the window (system / response / scratchpad /
/// code / memory …). The compiler reserves these before the free competition.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct RegionBudget {
    pub region: String,
    pub target_tokens: usize,
    pub max_tokens: usize,
}

/// The resolved reservation plan: how many tokens are carved for each
/// always-present region and how many are left to compete.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct Reservations {
    pub system: usize,
    pub response: usize,
    pub scratchpad: usize,
}

impl Reservations {
    pub fn total(&self) -> usize {
        self.system + self.response + self.scratchpad
    }
}

/// A token counter. Real tokenization when a `tokenizers::Tokenizer` is loaded,
/// else a deterministic `chars/4` estimate. Cheap to clone (Arc inside).
#[derive(Clone, Default)]
pub struct TokenCounter {
    inner: Arc<RwLock<Option<Tokenizer>>>,
}

impl TokenCounter {
    /// A counter with no tokenizer: uses the `chars/4` fallback. Deterministic
    /// and dependency-free — the default for tests and offline use.
    pub fn heuristic() -> Self {
        Self {
            inner: Arc::new(RwLock::new(None)),
        }
    }

    /// Load a tokenizer from a `tokenizer.json` file (the HuggingFace format
    /// `tokenizers` reads). Returns the counter on success.
    pub fn from_file(path: impl AsRef<std::path::Path>) -> Result<Self, String> {
        let tok = Tokenizer::from_file(path.as_ref()).map_err(|e| e.to_string())?;
        Ok(Self {
            inner: Arc::new(RwLock::new(Some(tok))),
        })
    }

    /// Build from an in-memory `tokenizer.json` blob.
    pub fn from_bytes(bytes: &[u8]) -> Result<Self, String> {
        let tok = Tokenizer::from_bytes(bytes).map_err(|e| e.to_string())?;
        Ok(Self {
            inner: Arc::new(RwLock::new(Some(tok))),
        })
    }

    /// True when a real tokenizer backs this counter (vs the fallback).
    pub fn is_accurate(&self) -> bool {
        self.inner.read().is_some()
    }

    /// Count tokens in `text`. Exact when a tokenizer is loaded; `chars/4`
    /// otherwise. The fallback is intentionally an *over*-estimate-safe round
    /// (`(chars+3)/4`) so the reserve invariant is never violated by undercount.
    pub fn count(&self, text: &str) -> usize {
        if let Some(tok) = self.inner.read().as_ref() {
            if let Ok(enc) = tok.encode(text, false) {
                return enc.len();
            }
        }
        estimate_tokens(text)
    }
}

/// Deterministic `chars/4` fallback token estimate (bible §4.2 fallback).
pub fn estimate_tokens(text: &str) -> usize {
    text.chars().count().saturating_add(3) / 4
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn heuristic_counter_uses_chars_over_four() {
        let c = TokenCounter::heuristic();
        assert!(!c.is_accurate());
        assert_eq!(c.count("abcd"), 1);
        assert_eq!(c.count("abcdefgh"), 2);
        assert_eq!(c.count(""), 0);
    }

    #[test]
    fn reservations_carve_before_fill() {
        let b = TokenBudget {
            max_input_tokens: 1000,
            reserve_output_tokens: 200,
            hard_limit_tokens: 1000,
        };
        assert_eq!(b.available_input(), 800);
        assert_eq!(b.reserve_pct(0.25), 200);
    }
}

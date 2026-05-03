//! Competitor-process spawning. One backend per file, all behind the
//! [`Competitor`] trait so the `competitive` suite can run them
//! through the same loop.
//!
//! Each backend captures its own version string (so the audit doc
//! can cite exact pinned versions), spawns a child process (or for
//! dismantle, drives the in-process Engine), and parses tok/s out
//! of the backend-specific output format.

pub mod dismantle;
pub mod llamacpp;
pub mod mlx;

pub use dismantle::DismantleBackend;
pub use llamacpp::LlamaCppBackend;
pub use mlx::MlxBackend;

use anyhow::Result;
use std::path::Path;

/// A single (backend, prompt) measurement.
#[derive(Debug, Clone, Default)]
pub struct Measurement {
    pub decode_tps: Option<f64>,
    pub prefill_tps: Option<f64>,
    pub ttft_ms: Option<f64>,
    pub peak_rss_mb: Option<f64>,
    pub output: String,
}

/// One backend in the head-to-head matrix.
pub trait Competitor: Send {
    /// Short identifier used in JSON output ("llamacpp", "dismantle").
    fn name(&self) -> &'static str;

    /// Pinned version string. Captured at construction time so the
    /// JSON output is reproducible against the same backend version.
    fn version(&self) -> String;

    /// Phase tag — only dismantle uses this (returns `Some(0)` until
    /// Phase 1 lands real Metal kernels). All competitors return `None`.
    fn phase_tag(&self) -> Option<u32> {
        None
    }

    /// Run one measurement on this prompt.
    fn run(
        &mut self,
        weights: &Path,
        prompt: &str,
        max_tokens: usize,
        temperature: f32,
    ) -> Result<Measurement>;
}

/// Parse a number out of an arbitrary output blob using a regex-ish
/// substring match. Helper used by the `llamacpp` stdout/stderr
/// parser — keeps the backend files small and tested.
pub(crate) fn extract_after(s: &str, marker: &str) -> Option<f64> {
    let pos = s.find(marker)?;
    let tail = &s[pos + marker.len()..];
    // Skip whitespace, '=', '(', etc.
    let tail = tail.trim_start_matches([' ', '=', '(', ':']);
    let mut end = 0usize;
    for (i, c) in tail.char_indices() {
        if c.is_ascii_digit() || c == '.' || c == '-' {
            end = i + c.len_utf8();
        } else {
            break;
        }
    }
    if end == 0 {
        return None;
    }
    tail[..end].parse::<f64>().ok()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn extract_after_handles_assignment() {
        assert_eq!(extract_after("foo = 42.5 bar", "foo"), Some(42.5));
    }

    #[test]
    fn extract_after_handles_paren() {
        let s = "eval time = 1234.5 ms / 256 runs (123.45 tokens per second)";
        assert_eq!(
            extract_after(s, "(").map(|x| (x * 100.0).round() / 100.0),
            Some(123.45)
        );
    }

    #[test]
    fn extract_after_returns_none_for_missing() {
        assert_eq!(extract_after("nothing here", "missing"), None);
    }
}

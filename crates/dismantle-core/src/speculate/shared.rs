//! Shared-expert draft + verify loop. The hot loop runs the shared
//! path forward N tokens, then runs routed experts in a single
//! batched pass to verify; accepted prefix is committed, the rest is
//! discarded and the verifier's first divergent token is taken.
//!
//! Lands in Phase 4.5. The structure here defines the data flow that
//! the model layer plugs into; the actual draft/verify plumbing stays
//! a stub until Phase 4.5.

use crate::Result;

#[derive(Debug, Clone, Default)]
pub struct DraftStats {
    pub draft_steps: usize,
    pub accepted: usize,
    pub rejected: usize,
}

/// One draft step: produce a candidate token id from the
/// shared-expert path, plus the per-vocab logits the verifier will
/// dot-check.
#[derive(Debug, Clone)]
pub struct DraftToken {
    pub id: u32,
    pub draft_logits: Vec<f32>,
}

/// Verify a window of draft tokens against the verifier's logits.
/// Returns the count of accepted tokens (longest agreeing prefix). On
/// disagreement, the verifier's first non-matching argmax is the
/// committed token.
pub fn verify_window(drafts: &[DraftToken], verifier_logits: &[Vec<f32>]) -> Result<usize> {
    if drafts.len() != verifier_logits.len() {
        return Err(crate::Error::Model("verify window length mismatch".into()));
    }
    let mut accepted = 0usize;
    for (d, v) in drafts.iter().zip(verifier_logits.iter()) {
        let v_argmax = argmax(v);
        if v_argmax == d.id {
            accepted += 1;
        } else {
            break;
        }
    }
    Ok(accepted)
}

fn argmax(xs: &[f32]) -> u32 {
    let mut best = 0usize;
    let mut bv = f32::NEG_INFINITY;
    for (i, &v) in xs.iter().enumerate() {
        if v > bv {
            best = i;
            bv = v;
        }
    }
    best as u32
}

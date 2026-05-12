//! Shared-expert draft + verify helpers.
//!
//! ExactShared is intentionally conservative: draft tokens come from the
//! shared-expert-only path, verifier tokens come from the full model, and the
//! accepted prefix is the longest greedy prefix where both agree. Low
//! acceptance makes this slower than normal decode, so callers should keep it
//! as an experimental path rather than a headline performance mode.

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

/// Result of verifying a window of draft tokens.
#[derive(Debug, Clone, Default)]
pub struct VerifyResult {
    /// Number of draft tokens accepted (longest agreeing prefix).
    pub accepted_count: usize,
    /// The verifier's argmax at the first point of disagreement.
    /// `None` if all draft tokens were accepted.
    pub first_divergent_token: Option<u32>,
}

/// Verify a window of draft tokens against the verifier's logits.
/// Returns a `VerifyResult` with the accepted prefix length and the
/// verifier's correction token (if any). On full agreement,
/// `first_divergent_token` is `None`.
pub fn verify_window(drafts: &[DraftToken], verifier_logits: &[Vec<f32>]) -> Result<VerifyResult> {
    if drafts.len() != verifier_logits.len() {
        return Err(crate::Error::Model("verify window length mismatch".into()));
    }
    let mut accepted_count = 0usize;
    let mut first_divergent_token = None;
    for (d, v) in drafts.iter().zip(verifier_logits.iter()) {
        let v_argmax = argmax(v);
        if v_argmax == d.id {
            accepted_count += 1;
        } else {
            first_divergent_token = Some(v_argmax);
            break;
        }
    }
    Ok(VerifyResult {
        accepted_count,
        first_divergent_token,
    })
}

/// Verify draft token ids with an argmax-producing verifier, stopping at the
/// first mismatch so low-acceptance windows do not pay for discarded verifier
/// work. On full agreement, callers still need one extra verifier step to emit
/// the bonus token after the accepted draft prefix.
pub fn verify_draft_ids_until_mismatch<F>(
    draft_ids: &[u32],
    mut verify_id: F,
) -> Result<VerifyResult>
where
    F: FnMut(usize) -> Result<u32>,
{
    let mut accepted_count = 0usize;
    let mut first_divergent_token = None;
    for (i, &draft_id) in draft_ids.iter().enumerate() {
        let verifier_id = verify_id(i)?;
        if verifier_id == draft_id {
            accepted_count += 1;
        } else {
            first_divergent_token = Some(verifier_id);
            break;
        }
    }
    Ok(VerifyResult {
        accepted_count,
        first_divergent_token,
    })
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn verify_ids_accepts_full_prefix() {
        let drafts = [3, 5, 8];
        let mut calls = 0usize;
        let result = verify_draft_ids_until_mismatch(&drafts, |i| {
            calls += 1;
            Ok(drafts[i])
        })
        .expect("verify ids");

        assert_eq!(result.accepted_count, 3);
        assert_eq!(result.first_divergent_token, None);
        assert_eq!(calls, 3);
    }

    #[test]
    fn verify_ids_stops_at_first_mismatch() {
        let drafts = [3, 5, 8, 13];
        let verifier = [3, 99, 8, 13];
        let mut calls = 0usize;
        let result = verify_draft_ids_until_mismatch(&drafts, |i| {
            calls += 1;
            Ok(verifier[i])
        })
        .expect("verify ids");

        assert_eq!(result.accepted_count, 1);
        assert_eq!(result.first_divergent_token, Some(99));
        assert_eq!(calls, 2);
    }
}
